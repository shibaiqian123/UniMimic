// GR00T reference 2f07b35b3f3b9053b1f4fd2adeaf56b6bce32366:
// g1_deploy_onnx_ref.cpp CheckMode/ReleaseMode startup; bounded retries here.
// This utility never creates a motor command publisher.
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <iostream>
#include <thread>
#include <chrono>
int main(int argc,char** argv) {
  if(argc!=2 || std::string(argv[1])=="lo" || std::string(argv[1]).empty())return 2;
  unitree::robot::ChannelFactory::Instance()->Init(0,argv[1]);
  unitree::robot::b2::MotionSwitcherClient client;
  client.SetTimeout(3.0f); client.Init();
  for(int attempt=0;attempt<4;++attempt) {
    std::string form,name;
    auto status=client.CheckMode(form,name);
    if(status!=0){std::cerr<<"CheckMode failed: "<<status<<std::endl;return 1;}
    if(name.empty()){std::cout<<"Motion service released"<<std::endl;return 0;}
    if(attempt==3)break;
    std::cout<<"Releasing active mode: "<<name<<std::endl;
    status=client.ReleaseMode();
    if(status!=0){std::cerr<<"ReleaseMode failed: "<<status<<std::endl;return 1;}
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  std::cerr<<"Motion service remains active after three release attempts"<<std::endl;
  return 1;
}
