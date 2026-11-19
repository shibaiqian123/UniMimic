#include "g1_sim_shm.h"
extern "C" {
void* fs_open(const char* path,int create){return open_sim(path,create!=0);}
void fs_close(void* h){if(h)munmap(h,sizeof(SimShared));}
int fs_state_size(){return sizeof(SimState);}
int fs_cmd_size(){return sizeof(SimCmd);}
void fs_write_state(void* h,const void* data){
 auto* s=(SimShared*)h;lock_sim(&s->state_mu);memcpy(&s->state,data,sizeof(SimState));pthread_mutex_unlock(&s->state_mu);
}
void fs_read_state(void* h,void* data){
 auto* s=(SimShared*)h;lock_sim(&s->state_mu);memcpy(data,&s->state,sizeof(SimState));pthread_mutex_unlock(&s->state_mu);
}
void fs_read_cmd(void* h,void* data){
 auto* s=(SimShared*)h;lock_sim(&s->cmd_mu);memcpy(data,&s->cmd,sizeof(SimCmd));pthread_mutex_unlock(&s->cmd_mu);
}
void fs_counts(void* h,uint64_t* out){
 auto* s=(SimShared*)h;
 out[0]=__atomic_load_n(&s->published,__ATOMIC_RELAXED);
 out[1]=__atomic_load_n(&s->received,__ATOMIC_RELAXED);
 out[2]=__atomic_load_n(&s->crc_bad,__ATOMIC_RELAXED);
 out[3]=__atomic_load_n(&s->write_fail,__ATOMIC_RELAXED);
}
}
