#pragma once
#include <stdint.h>
#include <pthread.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
// Simulation-side IPC only. This process is never allowed on DDS domain 0.
struct SimState {
 double sample_wall, sim_time;
 uint64_t tick, keys;
 double q[29], dq[29], tau[29], gyro[3], quat[4], acc[3], temp[58];
 double qpos[36], qvel[35]; // visualization/evaluation only; never encoded into policy inputs
 uint64_t support, motor_fault;
};
struct SimCmd {
 double received;
 uint64_t seq;
 double q[29], dq[29], kp[29], kd[29], tau[29];
 uint64_t mode[29], mode_machine, mode_pr;
};
struct SimShared {
 uint64_t magic;
 pthread_mutex_t state_mu, cmd_mu;
 SimState state;
 SimCmd cmd;
 uint64_t published, received, crc_bad, write_fail;
};
inline void lock_sim(pthread_mutex_t* p) {
 int rc=pthread_mutex_lock(p);
 if(rc==EOWNERDEAD)pthread_mutex_consistent(p);
}
inline SimShared* open_sim(const char* path,bool create) {
 int fd=open(path,create?(O_RDWR|O_CREAT|O_EXCL):O_RDWR,0600);
 if(fd<0)return nullptr;
 if(create && ftruncate(fd,sizeof(SimShared))!=0){close(fd);return nullptr;}
 struct stat st;
 if(fstat(fd,&st)!=0 || st.st_size!=sizeof(SimShared)){close(fd);return nullptr;}
 auto* p=(SimShared*)mmap(nullptr,sizeof(SimShared),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
 close(fd);if(p==MAP_FAILED)return nullptr;
 if(create){
  memset(p,0,sizeof(*p));
  pthread_mutexattr_t a;pthread_mutexattr_init(&a);
  pthread_mutexattr_setpshared(&a,PTHREAD_PROCESS_SHARED);
  pthread_mutexattr_setrobust(&a,PTHREAD_MUTEX_ROBUST);
  pthread_mutex_init(&p->state_mu,&a);pthread_mutex_init(&p->cmd_mu,&a);
  pthread_mutexattr_destroy(&a);p->magic=0x424C494E4B53494DULL;
 } else if(p->magic!=0x424C494E4B53494DULL){munmap(p,sizeof(*p));return nullptr;}
 return p;
}
