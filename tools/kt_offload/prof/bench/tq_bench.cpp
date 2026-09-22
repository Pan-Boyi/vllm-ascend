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
  return 0;
}
