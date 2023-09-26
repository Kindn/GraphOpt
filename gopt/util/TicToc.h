#ifndef _TIC_TOC_H_
#define _TIC_TOC_H_

#include <ctime>
#include <cstdlib>
#include <chrono>

namespace gopt {

class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::steady_clock::now();
    }

    double toc()
    {
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = 
          std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        return elapsed_seconds.count();
    }

  private:
    std::chrono::steady_clock::time_point start, end;
};

}

#endif // _TIC_TOC_H_