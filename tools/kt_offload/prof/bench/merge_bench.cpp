#include <immintrin.h>
#include <numa.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
const int H=7168;
static void merge(float*a,float*b,unsigned short*out){
  for(int e=0;e<H;e+=16){
    __m512 x=_mm512_add_ps(_mm512_loadu_ps(a+e),_mm512_loadu_ps(b+e));
    _mm512_storeu_ps(a+e,x);
  }
  for(int e=0;e<H;e+=16){
    __m512 x=_mm512_loadu_ps(a+e);
    __m256i y=_mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(x),16));
    _mm256_storeu_si256((__m256i*)(out+e),y);
  }
}
int main(int argc,char**argv){
  int remote = argc>1;
  float*a=(float*)numa_alloc_onnode(H*4,0);
  float*b=(float*)numa_alloc_onnode(H*4, remote?1:0);
  unsigned short*o=(unsigned short*)numa_alloc_onnode(H*2,0);
  for(int i=0;i<H;i++){a[i]=1;b[i]=2;}
  std::vector<double> v;
  for(int i=0;i<20000;i++){
    // flush b from cache to mimic cold cross-numa read
    for(int e=0;e<H;e+=16) _mm_clflushopt((char*)(b+e));
    _mm_sfence();
    auto s=std::chrono::steady_clock::now();
    merge(a,b,o);
    auto t=std::chrono::steady_clock::now();
    v.push_back(std::chrono::duration<double,std::micro>(t-s).count());
  }
  std::sort(v.begin(),v.end());
  printf("%s merge(7168) us: med %.2f p90 %.2f\n", remote?"REMOTE b":"LOCAL  b", v[v.size()/2],v[v.size()*9/10]);
}
