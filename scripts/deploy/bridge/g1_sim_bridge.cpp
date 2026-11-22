// Simulation-side Unitree DDS adapter. Domain 1/2 and loopback only.
#include "g1_sim_shm.h"
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <signal.h>
#include <time.h>
#include <stdio.h>
#include <cmath>
#include <string>
#include <sys/prctl.h>
#include <sched.h>
#include <cstdlib>
static unsigned long long rqwait(){FILE* f=fopen("/proc/thread-self/schedstat","r");unsigned long long a=0,b=0,c=0;if(f){if(fscanf(f,"%llu %llu %llu",&a,&b,&c)!=3)b=0;fclose(f);}return b;}
using namespace unitree::robot;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;
static SimShared* shared=nullptr;
static volatile sig_atomic_t stop=0;
static double mono(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void signal_stop(int){stop=1;}
// Copied from unitree_sdk2 (https://github.com/unitreerobotics/unitree_sdk2),
// example/g1/low_level/g1_ankle_swing_example.cpp.
// Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"), BSD-3-Clause;
// see LICENSES/unitree_sdk2-BSD-3-Clause.txt and LICENSES/THIRD_PARTY_NOTICES.md.
static uint32_t Crc32Core(const uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0, data = 0, CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1u << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }
      if (data & xbit) CRC32 ^= dwPolynomial;
      xbit >>= 1;
    }
  }
  return CRC32;
}

static void command(const void* data){
 const auto& m=*(const LowCmd_*)data;
 if(m.crc()!=Crc32Core((const uint32_t*)&m,(sizeof(m)>>2)-1)){
  __atomic_add_fetch(&shared->crc_bad,1,__ATOMIC_RELAXED);return;
 }
 SimCmd c{};c.received=mono();
 c.seq=__atomic_add_fetch(&shared->received,1,__ATOMIC_RELAXED);
 c.mode_machine=m.mode_machine();c.mode_pr=m.mode_pr();
 for(int i=0;i<29;i++){
  const auto& x=m.motor_cmd()[i];c.q[i]=x.q();c.dq[i]=x.dq();c.kp[i]=x.kp();c.kd[i]=x.kd();c.tau[i]=x.tau();c.mode[i]=x.mode();
 }
 lock_sim(&shared->cmd_mu);shared->cmd=c;pthread_mutex_unlock(&shared->cmd_mu);
}
int main(int argc,char** argv){
 if(argc!=3){fprintf(stderr,"usage: g1_sim_bridge <session.mmap> <domain 1|2>\n");return 2;}
 int domain=std::stoi(argv[2]);
 if(domain!=1 && domain!=2){fprintf(stderr,"simulation refuses hardware domain\n");return 2;}
 shared=open_sim(argv[1],false);if(!shared)return 3;
 signal(SIGINT,signal_stop);signal(SIGTERM,signal_stop);
 ChannelFactory::Instance()->Init(domain,"lo");
 ChannelSubscriber<LowCmd_> sub("rt/lowcmd");sub.InitChannel(command,0);
 ChannelPublisher<LowState_> pub("rt/lowstate");pub.InitChannel();
 double deadline=mono(),last_publish=0.;
 double last_loop_cpu=0.;unsigned long long before_wait=0;
 const bool trace_sched=getenv("BLINK_DDS_TRACE_SCHED")!=nullptr;
 fprintf(stderr,"publisher scheduler=%d timerslack_ns=%d\n",sched_getscheduler(0),prctl(PR_GET_TIMERSLACK));
 while(!stop){
  double loop_start=mono();timespec cpu_ts;clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu_ts);
  double loop_cpu=cpu_ts.tv_sec+cpu_ts.tv_nsec*1e-9;
  SimState s{};lock_sim(&shared->state_mu);s=shared->state;pthread_mutex_unlock(&shared->state_mu);
  double read_end=mono();
  if(s.sample_wall>0){
   LowState_ m{};m.mode_machine()=5;m.mode_pr()=0;m.tick()=uint32_t(s.tick);
   for(int i=0;i<29;i++){
    auto& x=m.motor_state()[i];x.q()=s.q[i];x.dq()=s.dq[i];x.tau_est()=s.tau[i];
    x.temperature()[0]=int16_t(s.temp[2*i]);x.temperature()[1]=int16_t(s.temp[2*i+1]);x.motorstate()=uint32_t(s.motor_fault);
   }
   for(int i=0;i<4;i++)m.imu_state().quaternion()[i]=s.quat[i];
   for(int i=0;i<3;i++){m.imu_state().gyroscope()[i]=s.gyro[i];m.imu_state().accelerometer()[i]=s.acc[i];}
   double w=s.quat[0],x=s.quat[1],y=s.quat[2],z=s.quat[3];
   m.imu_state().rpy()[0]=atan2(2*(w*x+y*z),1-2*(x*x+y*y));
   m.imu_state().rpy()[1]=asin(fmax(-1.,fmin(1.,2*(w*y-z*x))));
   m.imu_state().rpy()[2]=atan2(2*(w*z+x*y),1-2*(y*y+z*z));
   m.wireless_remote()[2]=s.keys&255;m.wireless_remote()[3]=(s.keys>>8)&255;
   m.crc()=Crc32Core((const uint32_t*)&m,(sizeof(m)>>2)-1);
   double before=mono();
   bool ok=false;try{ok=pub.Write(m);}catch(...){}
   double after=mono();
   if(last_publish>0 && after-last_publish>.01)
    fprintf(stderr,"publish_gap t=%.9f gap_ms=%.3f write_ms=%.3f sample_age_ms=%.3f tick=%llu wake_late_ms=%.3f read_ms=%.3f build_ms=%.3f loop_cpu_ms=%.3f runqueue_wait_ms=%.3f\n",after,(after-last_publish)*1000,(after-before)*1000,(before-s.sample_wall)*1000,(unsigned long long)s.tick,(loop_start-deadline)*1000,(read_end-loop_start)*1000,(before-read_end)*1000,(loop_cpu-last_loop_cpu)*1000,trace_sched ? (rqwait()-before_wait)/1e6 : -1.);
   last_publish=after;
   __atomic_add_fetch(&shared->published,1,__ATOMIC_RELAXED);
   if(!ok)__atomic_add_fetch(&shared->write_fail,1,__ATOMIC_RELAXED);
  }
  last_loop_cpu=loop_cpu;
  deadline+=.001;double now=mono();
  if(deadline<now-.01)deadline=now;
  if(trace_sched)before_wait=rqwait();
  if(deadline>now){timespec ts{time_t(deadline),long((deadline-time_t(deadline))*1e9)};clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&ts,nullptr);}
 }
 munmap(shared,sizeof(*shared));return 0;
}
