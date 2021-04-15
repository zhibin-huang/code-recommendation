#include <thread>
#include <iostream>
using namespace std;

void a(int i){
    this_thread::sleep_for(std::chrono::seconds(i));
    cout << i <<endl;
}

int main(){
    std::thread t[30];
    for(int i = 0; i < 30; ++i){
        t[i] = std::thread(a, i + 1);
    }
    cout << "wating..." <<endl;
    for(int i = 0; i < 30; ++i){
        t[i].join();
    }
    cout << "done!\n";
    return 0;
}