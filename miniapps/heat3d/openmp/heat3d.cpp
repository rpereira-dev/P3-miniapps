#include <iostream>
#include <chrono>
#include <array>
#include "types.hpp"
#include "Config.hpp"
#include "../Parser.hpp"
#include "Init.hpp"
#include "IO.hpp"
#include "Timestep.hpp"

# include "energy.h"

int main(int argc, char *argv[]) {

  #pragma omp target
  {
      if (!omp_is_initial_device())
          printf("Hello world from accelerator.\n");
      else
          printf("Hello world from host.\n");
  }

  Parser parser(argc, argv);
  auto shape = parser.shape_;
  int nbiter = parser.nbiter_;
  int freq_diag = parser.freq_diag_;
  bool enable_diag = freq_diag > 0;
  int nx = shape[0], ny = shape[1], nz = shape[2];

  Config conf(nx, ny, nz, nbiter, freq_diag);

  std::vector<Timer*> timers;
  defineTimers(timers);

  RealView1D x, y, z;
  RealView3D u, un;

  initialize(conf, x, y, z, u, un);
  power_init();
  power_start();

  // Main loop
  timers[Total]->begin();
  for(int i=0; i<conf.nbiter; i++) {
    timers[MainLoop]->begin();
    if(enable_diag) to_csv(conf, u, i, timers);
    step(conf, u, un, timers);
    u.swap(un);
    timers[MainLoop]->end();
  }
  timers[Total]->end();

  using real_type = RealView3D::value_type;
  real_type time = conf.dt * conf.nbiter;
  finalize(conf, time, x, y, z, u, un);
  
  // Measure performance
  double Gflops = performance(conf, timers[Total]->seconds());
  double P = power_stop();
  printf("Gflop/s/watt = %lf\n", Gflops/P);
  
  printTimers(timers);
  freeTimers(timers);

  power_deinit();


  return 0;
}
