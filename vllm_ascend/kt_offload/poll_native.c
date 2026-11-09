/* SPDX-License-Identifier: Apache-2.0
 *
 * P1b: the one piece of this mechanism that cannot be written in Python.
 *
 * The graph signals the host by writing a step number into a pinned word (an
 * 8-byte D2H enqueued behind this layer's input copies on the same stream).
 * Something has to watch those words and react in microseconds, and whatever
 * watches them must NOT hold the GIL while it does: a Python thread spinning
 * on ``word.item()`` executes bytecode, so it holds the GIL between checks and
 * the main thread pays up to ``sys.getswitchinterval()`` (5 ms by default) to
 * get it back -- on a 30 ms TPOT with 40 layers that is not a tax, it is the
 * whole budget.
 *
 * ``ctypes.CDLL`` releases the GIL for the duration of a foreign call, so the
 * spin below runs with the GIL free and the interpreter keeps running. That is
 * the only reason this file exists; everything else about the protocol
 * (expected step numbers, which layer, what to run) stays in Python where it
 * can be read and tested.
 *
 * 🔴 It is NOT in kt-kernel on purpose. Putting it there would mean a new
 * wheel, a new site directory and a re-run of t0/t1/t2 for a 60-line spin
 * loop that needs nothing kt has. See docs and RUNBOOK_p1b.md §0.
 *
 * What the loop may and may not assume
 * ------------------------------------
 * * The word is written by device DMA into pinned (page-locked) host memory,
 *   so the read has to be an acquire load of a volatile location, never a
 *   value the compiler may hoist out of the loop.
 * * Each word sits on its own 128-byte stride (``stride_words``): the entries
 *   of one scan are read by this thread while the device writes one of them,
 *   and sharing a line between two of them would mean a DMA write invalidating
 *   a line this loop is spinning on for a different layer.
 * * A step counter only ever grows, so "the signal arrived" is ``word >=
 *   expected``, not ``==``. The equality -- i.e. "no step was missed" -- is
 *   judged in Python, because a mismatch is a diagnosis, not a wait condition.
 * * ``expected[i] == 0`` means "this slot is not in use"; step numbers start at
 *   1 (the counter starts at 0 and the first replay increments it). Without
 *   that sentinel an unused slot would match immediately on every scan.
 */

#include <sched.h>
#include <stdint.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define KT_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define KT_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
#define KT_PAUSE() __asm__ __volatile__("" ::: "memory")
#endif

/* Scans between two clock reads. clock_gettime is ~25 ns here (vDSO), and a
 * scan of a few dozen words is ~100 ns, so reading the clock every time would
 * be a fifth of the loop. 64 scans is at most a few microseconds of overshoot
 * on the timeout, which nothing here cares about. */
#define KT_CLOCK_EVERY 64

static inline uint64_t kt_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Wait until one of ``n`` signal words has reached its expected step.
 *
 *   words        base of the pinned signal array
 *   expected     expected[i] = the step number slot i is waiting for
 *                (0 = slot unused, never matches)
 *   stride_words how many uint64 apart two consecutive slots are (padding)
 *   timeout_ns   give up and return -1 after this long (<=0: never)
 *   hot_ns       spin without sleeping for this long after entry; afterwards
 *                sleep ``sleep_ns`` between scans. 0 = sleep from the start.
 *   sleep_ns     the sleep in the cold phase (<=0: sched_yield instead)
 *   out          optional uint64[2]: out[0] = full scans made, out[1] = the
 *                CLOCK_MONOTONIC nanosecond at which the match was seen.
 *                out[1] is taken HERE, before returning, because the caller
 *                cannot take it: a Python thread coming back from this call
 *                first has to acquire the GIL, and that wait is not part of
 *                what this loop costs. It is the only way to measure the
 *                wake-up latency of the spin itself.
 *
 * Returns the slot index that is ready, or -1 on timeout.
 */
long kt_wait_signal(const volatile uint64_t *words, const uint64_t *expected, long n, long stride_words,
                    long timeout_ns, long hot_ns, long sleep_ns, uint64_t *out) {
    uint64_t t0 = kt_now_ns();
    uint64_t scans = 0;
    long since_clock = 0;
    int hot = hot_ns > 0;
    struct timespec nap;
    nap.tv_sec = 0;
    nap.tv_nsec = sleep_ns > 0 ? sleep_ns : 0;

    for (;;) {
        for (long i = 0; i < n; ++i) {
            uint64_t want = expected[i];
            if (want == 0) {
                continue;
            }
            uint64_t got = __atomic_load_n((const uint64_t *)(words + (size_t)i * (size_t)stride_words),
                                           __ATOMIC_ACQUIRE);
            if (got >= want) {
                if (out) {
                    out[0] = scans;
                    out[1] = kt_now_ns();
                }
                return i;
            }
        }
        ++scans;
        if (hot) {
            KT_PAUSE();
        } else if (sleep_ns > 0) {
            nanosleep(&nap, NULL);
        } else {
            sched_yield();
        }
        if (++since_clock >= KT_CLOCK_EVERY) {
            since_clock = 0;
            uint64_t dt = kt_now_ns() - t0;
            if (hot && (uint64_t)hot_ns <= dt) {
                hot = 0;
            }
            if (timeout_ns > 0 && dt >= (uint64_t)timeout_ns) {
                if (out) {
                    out[0] = scans;
                    out[1] = t0 + dt;
                }
                return -1;
            }
        }
    }
}

/* One acquire load of slot i. Python reads the same words through the torch
 * tensor; this exists so a test can prove the two agree, and so a diagnosis
 * ("what did the word actually hold?") reads memory the same way the wait
 * does rather than through a different path. */
uint64_t kt_read_signal(const volatile uint64_t *words, long i, long stride_words) {
    return __atomic_load_n((const uint64_t *)(words + (size_t)i * (size_t)stride_words), __ATOMIC_ACQUIRE);
}

/* Write ``value`` into slot i after ``delay_ns``, and report when. Used by the
 * tests and by nothing else: it is how "another agent writes the word while
 * the loop spins" is driven without the GIL anywhere near either side, which
 * is what makes the measured wake-up latency the loop's own. The device does
 * the same thing with a DMA. */
uint64_t kt_poke_after(volatile uint64_t *words, long i, long stride_words, uint64_t value, long delay_ns) {
    if (delay_ns > 0) {
        struct timespec nap;
        nap.tv_sec = delay_ns / 1000000000L;
        nap.tv_nsec = delay_ns % 1000000000L;
        nanosleep(&nap, NULL);
    }
    uint64_t now = kt_now_ns();
    __atomic_store_n((uint64_t *)(words + (size_t)i * (size_t)stride_words), value, __ATOMIC_RELEASE);
    return now;
}

/* Build stamp, so a loaded .so can be matched to this source. */
long kt_poll_abi(void) { return 3; }
