// kt's TaskQueue round trip, and what P2 removes.
//
// §4.22 F4: CPUInfer::submit hands the task to the TaskQueue worker thread and
// sync() waits for its notify -- a full thread hand-off per MoE layer, 40 per
// token. This measures it against running the same task on the calling thread
// (TaskQueue::run_inline, §4.22 P2), which is what the inline path does.
//
// Build and run (kt-kernel tree; nothing but the queue is needed):
//   g++ -std=c++20 -O2 -I<kt>/cpu_backend tools/kt_offload/prof/bench/tq_bench.cpp \
//       <kt>/cpu_backend/task_queue.cpp -o /tmp/tq_bench && /tmp/tq_bench
// The inline arm needs a kt that carries run_inline(); an older tree only
// builds the first two arms (delete the third).

#include "task_queue.h"
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
int main(){
  TaskQueue q;
  volatile int x=0;
  // warm
  for(int i=0;i<1000;i++){ q.enqueue([&]{x++;}); q.sync(0);}
  std::vector<double> v;
  for(int i=0;i<20000;i++){
    auto a=std::chrono::steady_clock::now();
    q.enqueue([&]{x++;});
    q.sync(0);
    auto b=std::chrono::steady_clock::now();
    v.push_back(std::chrono::duration<double,std::micro>(b-a).count());
  }
  std::sort(v.begin(),v.end());
  printf("empty-task submit+sync roundtrip us: med %.2f p10 %.2f p90 %.2f p99 %.2f\n",
     v[v.size()/2], v[v.size()/10], v[v.size()*9/10], v[v.size()*99/100]);
  // inline reference
  v.clear();
  for(int i=0;i<20000;i++){
    auto a=std::chrono::steady_clock::now();
    x++;
    auto b=std::chrono::steady_clock::now();
    v.push_back(std::chrono::duration<double,std::micro>(b-a).count());
  }
  std::sort(v.begin(),v.end());
  printf("inline reference us: med %.3f\n", v[v.size()/2]);
  // what P2 actually runs: the same task, on this thread, plus the sync that
  // still has to be called (it takes the mutex and returns: pending is 0)
  v.clear();
  for(int i=0;i<1000;i++){ q.run_inline([&]{x++;}); q.sync(0);}
  for(int i=0;i<20000;i++){
    auto a=std::chrono::steady_clock::now();
    q.run_inline([&]{x++;});
    q.sync(0);
    auto b=std::chrono::steady_clock::now();
    v.push_back(std::chrono::duration<double,std::micro>(b-a).count());
  }
  std::sort(v.begin(),v.end());
  printf("run_inline+sync roundtrip us: med %.3f p10 %.3f p90 %.3f p99 %.3f\n",
     v[v.size()/2], v[v.size()/10], v[v.size()*9/10], v[v.size()*99/100]);
  return 0;
}
