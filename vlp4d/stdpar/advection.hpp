#ifndef __ADVECTION_HPP__
#define __ADVECTION_HPP__

#include "config.hpp"
#include "efield.hpp"
#include "types.hpp"
#include "Parallel_For.hpp"

namespace Advection {

#ifndef LAG_ORDER
#define LAG_ORDER 5
#endif

#if LAG_ORDER == 4
#define LAG_OFFSET 2
#define LAG_HALO_PTS 3
#define LAG_PTS 5

static inline void lag_basis(double posx, double * coef) {
  const double loc[] = {1./24., -1./6., 1./4., -1./6., 1./24.};

  coef[0] = loc[0] * (posx - 1.) * (posx - 2.) * (posx - 3.) * (posx - 4.);
  coef[1] = loc[1] * (posx) * (posx - 2.) * (posx - 3.) * (posx - 4.);
  coef[2] = loc[2] * (posx) * (posx - 1.) * (posx - 3.) * (posx - 4.);
  coef[3] = loc[3] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 4.);
  coef[4] = loc[4] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 3.);
}
#endif

#if LAG_ORDER == 5
#define LAG_OFFSET 2
#define LAG_HALO_PTS 3
#define LAG_PTS 6
#define LAG_ODD

static inline void lag_basis(double px, double * coef) {
  const double loc[] = { -1. / 24, 1. / 24., -1. / 12., 1. / 12., -1. / 24., 1. / 24. };
  const double pxm2 = px - 2.;
  const double sqrpxm2 = pxm2 * pxm2;
  const double pxm2_01 = pxm2 * (pxm2 - 1.);
  coef[0] = loc[0] * pxm2_01 * (pxm2 + 1.) * (pxm2 - 2.) * (pxm2 - 1.);
  coef[1] = loc[1] * pxm2_01 * (pxm2 - 2.) * (5 * sqrpxm2 + pxm2 - 8.);
  coef[2] = loc[2] * (pxm2 - 1.) * (pxm2 - 2.) * (pxm2 + 1.) * (5 * sqrpxm2 - 3 * pxm2 - 6.);
  coef[3] = loc[3] * pxm2 * (pxm2 + 1.) * (pxm2 - 2.) * (5 * sqrpxm2 - 7 * pxm2 - 4.);
  coef[4] = loc[4] * pxm2_01 * (pxm2 + 1.) * (5 * sqrpxm2 - 11 * pxm2 - 2.);
  coef[5] = loc[5] * pxm2_01 * pxm2 * (pxm2 + 1.) * (pxm2 - 2.);
}
#endif

#if LAG_ORDER == 7
#define LAG_OFFSET 3
#define LAG_HALO_PTS 4
#define LAG_PTS 7
#define LAG_ODD

static inline void lag_basis(const double px, double * coef) {
  const double loc[] = {-1. / 720, 1. / 720, -1. / 240, 1. / 144, -1. / 144, 1. / 240, -1. / 720, 1. / 720};
  const double pxm3 = px - 3.;
  const double sevenpxm3sqr = 7. * pxm3 * pxm3;
  const double f1t3 = (px - 1.) * (px - 2.) * (px - 3.);
  const double f4t6 = (px - 4.) * (px - 5.) * (px - 6.);
  coef[0] = loc[0] * f1t3 * f4t6 * (px - 4.);
  coef[1] = loc[1] * (px - 2.) * pxm3 * f4t6 * (sevenpxm3sqr + 8. * pxm3 - 18.);
  coef[2] = loc[2] * (px - 1.) * pxm3 * f4t6 * (sevenpxm3sqr + 2. * pxm3 - 15.);
  coef[3] = loc[3] * (px - 1.) * (px - 2.) * f4t6 * (sevenpxm3sqr - 4. * pxm3 - 12.);
  coef[4] = loc[4] * f1t3 * (px - 5.) * (px - 6.) * (sevenpxm3sqr - 10. * pxm3 - 9.);
  coef[5] = loc[5] * f1t3 * (px - 4.) * (px - 6.) * (sevenpxm3sqr - 16. * pxm3 - 6.);
  coef[6] = loc[6] * f1t3 * (px - 4.) * (px - 5.) * (sevenpxm3sqr - 22. * pxm3 - 3.);
  coef[7] = loc[7] * f1t3 * f4t6 * pxm3;
}
#endif

  struct advect_1D_x_functor {
    using mdspan_type = RealView4D::mdspan_type;
    Config* conf_;
    mdspan_type fn_, fnp1_;

    float64 dt_;
    int s_nxmax, s_nymax, s_nvxmax, s_nvymax;
    float64 hmin, hwidth, dh, idh;
    float64 minPhix, minPhivx, dx, dvx;

    advect_1D_x_functor(Config*conf, const RealView4D &fn, RealView4D &fnp1, float64 dt)
      : conf_(conf), dt_(dt) {
      const Domain *dom = &(conf_->dom_);
      s_nxmax  = dom->nxmax_[0];
      s_nymax  = dom->nxmax_[1];
      s_nvxmax = dom->nxmax_[2];
      s_nvymax = dom->nxmax_[3];

      hmin = dom->minPhy_[0];
      hwidth = dom->maxPhy_[0] - dom->minPhy_[0];
      dh = dom->dx_[0];
      idh = 1. / dom->dx_[0];

      minPhix  = dom->minPhy_[0];
      minPhivx = dom->minPhy_[2];
      dx       = dom->dx_[0];
      dvx      = dom->dx_[2];

      fn_   = fn.mdspan();
      fnp1_ = fnp1.mdspan();
    }

    // 3DRange policy over x, y, vx directions and sequential loop along vy direction
    void operator()(const int ix, const int iy, const int ivx) const {
      const double x  = minPhix + ix * dx;
      const double vx = minPhivx + ivx * dvx;
      const double depx = dt_ * vx;
      const double xstar = hmin + fmod(hwidth + x - depx - hmin, hwidth);
    
      double coef[LAG_PTS];
    
#ifdef LAG_ODD
      int ipos1 = floor((xstar-hmin) * idh);
#else
      int ipos1 = round((xstar-hmin) * idh);
#endif
      const double d_prev1 = LAG_OFFSET + idh * (xstar - (hmin + ipos1 *dh));
      ipos1 -= LAG_OFFSET;
      lag_basis(d_prev1, coef);
    
      SIMD_LOOP
      for(int ivy=0; ivy<s_nvymax; ivy++) {
        double ftmp = 0.;
        for(int k=0; k<=LAG_ORDER; k++) {
          int idx_ipos1 = (s_nxmax + ipos1 + k) % s_nxmax;
          ftmp += coef[k] * fn_(idx_ipos1, iy, ivx, ivy);
        }
        fnp1_(ix, iy, ivx, ivy) = ftmp;
      }
    }
  };

  struct advect_1D_y_functor {
    using mdspan_type = RealView4D::mdspan_type;
    Config* conf_;
    mdspan_type fn_, fnp1_;
    float64 dt_;
    int s_nxmax, s_nymax, s_nvxmax, s_nvymax;
    float64 hmin, hwidth, dh, idh;
    float64 minPhiy, minPhivy, dy, dvy;

    advect_1D_y_functor(Config* conf, const RealView4D &fn, RealView4D &fnp1, float64 dt)
      : conf_(conf), dt_(dt) {
      const Domain *dom = &(conf->dom_);
      s_nxmax  = dom->nxmax_[0];
      s_nymax  = dom->nxmax_[1];
      s_nvxmax = dom->nxmax_[2];
      s_nvymax = dom->nxmax_[3];

      hmin = dom->minPhy_[1];
      hwidth = dom->maxPhy_[1] - dom->minPhy_[1];
      dh = dom->dx_[1];
      idh = 1. / dom->dx_[1];

      minPhiy  = dom->minPhy_[1];
      minPhivy = dom->minPhy_[3];
      dy       = dom->dx_[1];
      dvy      = dom->dx_[3];

      fn_   = fn.mdspan();
      fnp1_ = fnp1.mdspan();
    }

    // 3DRange policy over x, y, vx directions and sequential loop along vy direction
    void operator()(const int ix, const int iy, const int ivx) const {
      const double y  = minPhiy + iy * dy;
    
      SIMD_LOOP
      for(int ivy=0; ivy<s_nvymax; ivy++) {
        const double vy = minPhivy + ivy * dvy;
        const double deph = dt_ * vy;
        
        const double hstar = hmin + fmod(hwidth + y - deph - hmin, hwidth);
        double coef[LAG_PTS];
         
#ifdef LAG_ODD
        int ipos1 = floor((hstar-hmin) * idh);
#else
        int ipos1 = round((hstar-hmin) * idh);
#endif
        const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 *dh));
        ipos1 -= LAG_OFFSET;
        lag_basis(d_prev1, coef);
        
        double ftmp = 0.;
        for(int k=0; k<=LAG_ORDER; k++) {
          int idx_ipos1 = (s_nymax + ipos1 + k) % s_nymax;
          ftmp += coef[k] * fn_(ix, idx_ipos1, ivx, ivy);
        }
        fnp1_(ix, iy, ivx, ivy) = ftmp;
      }
    }
  };

  struct advect_1D_vx_functor {
    using mdspan_2d_type = RealView2D::mdspan_type;
    using mdspan_4d_type = RealView4D::mdspan_type;
    Config* conf_;
    mdspan_4d_type fn_, fnp1_;
    Efield* ef_;

    mdspan_2d_type ex_;
    float64 dt_;
    int s_nxmax, s_nymax, s_nvxmax, s_nvymax;
    float64 hmin, hwidth, dh, idh;
    float64 minPhivx, dvx;

    advect_1D_vx_functor(Config* conf, const RealView4D &fn, RealView4D &fnp1, Efield* ef, float64 dt)
      : conf_(conf), ef_(ef), dt_(dt) {
      const Domain *dom = &(conf_->dom_);
      s_nxmax  = dom->nxmax_[0];
      s_nymax  = dom->nxmax_[1];
      s_nvxmax = dom->nxmax_[2];
      s_nvymax = dom->nxmax_[3];

      hmin = dom->minPhy_[2];
      hwidth = dom->maxPhy_[2] - dom->minPhy_[2];
      dh = dom->dx_[2];
      idh = 1. / dom->dx_[2];

      minPhivx = dom->minPhy_[2];
      dvx      = dom->dx_[2];

      ex_   = ef_->ex_.mdspan();
      fn_   = fn.mdspan();
      fnp1_ = fnp1.mdspan();
    }

    // 3DRange policy over x, y, vx directions and sequential loop along vy direction
    void operator()(const int ix, const int iy, const int ivx) const {
      const double vx = minPhivx + ivx * dvx;
      const double depx = dt_ * ex_(ix, iy);
      const double hstar = hmin + fmod(hwidth + vx - depx - hmin, hwidth);
    
      double coef[LAG_PTS];
    
#ifdef LAG_ODD
      int ipos1 = floor((hstar-hmin) * idh);
#else
      int ipos1 = round((hstar-hmin) * idh);
#endif
      const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 *dh));
      ipos1 -= LAG_OFFSET;
      lag_basis(d_prev1, coef);
    
      SIMD_LOOP
      for(int ivy=0; ivy<s_nvymax; ivy++) {
        double ftmp = 0.;
        for(int k=0; k<=LAG_ORDER; k++) {
          int idx_ipos1 = (s_nvxmax + ipos1 + k) % s_nvxmax;
          ftmp += coef[k] * fn_(ix, iy, idx_ipos1, ivy);
        }
        fnp1_(ix, iy, ivx, ivy) = ftmp;
      }
    }
  };

  struct advect_1D_vy_functor {
    using mdspan_2d_type = RealView2D::mdspan_type;
    using mdspan_4d_type = RealView4D::mdspan_type;
    Config* conf_;
    mdspan_4d_type fn_, fnp1_;
    Efield* ef_;

    mdspan_2d_type ey_;
    float64 dt_;
    int s_nxmax, s_nymax, s_nvxmax, s_nvymax;
    float64 hmin, hwidth, dh, idh;
    float64 minPhivy, dvy;

    advect_1D_vy_functor(Config* conf, const RealView4D &fn, RealView4D &fnp1, Efield* ef, float64 dt)
      : conf_(conf), ef_(ef), dt_(dt) {
      const Domain *dom = &(conf_->dom_);
      s_nxmax  = dom->nxmax_[0];
      s_nymax  = dom->nxmax_[1];
      s_nvxmax = dom->nxmax_[2];
      s_nvymax = dom->nxmax_[3];

      hmin = dom->minPhy_[3];
      hwidth = dom->maxPhy_[3] - dom->minPhy_[3];
      dh = dom->dx_[3];
      idh = 1. / dom->dx_[3];

      minPhivy = dom->minPhy_[3];
      dvy      = dom->dx_[3];

      ey_   = ef_->ey_.mdspan();
      fn_   = fn.mdspan();
      fnp1_ = fnp1.mdspan();
    }

    // 3DRange policy over x, y, vx directions and sequential loop along vy direction
    void operator()(const int ix, const int iy, const int ivx) const {
      for(int ivy=0; ivy<s_nvymax; ivy++) {
        const double vy = minPhivy + ivy * dvy;
        const double deph = dt_ * ey_(ix, iy);
        const double hstar = hmin + fmod(hwidth + vy - deph - hmin, hwidth);
    
        double coef[LAG_PTS];
    
#ifdef LAG_ODD
        int ipos1 = floor((hstar-hmin) * idh);
#else
        int ipos1 = round((hstar-hmin) * idh);
#endif
        const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 *dh));
        ipos1 -= LAG_OFFSET;
        lag_basis(d_prev1, coef);
    
        double ftmp = 0.;
        for(int k=0; k<=LAG_ORDER; k++) {
          int idx_ipos1 = (s_nvymax + ipos1 + k) % s_nvymax;
          ftmp += coef[k] * fn_(ix, iy, ivx, idx_ipos1);
        }
        fnp1_(ix, iy, ivx, ivy) = ftmp;
      }
    }
  };

  void advect_1D_x(Config *conf, const RealView4D &fn, RealView4D &fnp1, float64 dt);
  void advect_1D_y(Config *conf, const RealView4D &fn, RealView4D &fnp1, float64 dt);
  void advect_1D_vx(Config *conf, const RealView4D &fn, RealView4D &fnp1, Efield *ef, float64 dt);
  void advect_1D_vy(Config *conf, const RealView4D &fn, RealView4D &fnp1, Efield *ef, float64 dt);
  void print_fxvx(Config *conf, const RealView4D &fn, int iter);

  void advect_1D_x(Config *conf, const RealView4D &fn, RealView4D &fnp1, float64 dt) {
    const Domain *dom = &(conf->dom_);
    int s_nxmax  = dom->nxmax_[0];
    int s_nymax  = dom->nxmax_[1];
    int s_nvxmax = dom->nxmax_[2];

    Iterate_policy<3> policy3d({0, 0, 0}, {s_nxmax,s_nymax,s_nvxmax});
    Impl::for_each(policy3d, advect_1D_x_functor(conf, fn, fnp1, dt));
  };

  void advect_1D_y(Config *conf, const RealView4D &fn, RealView4D &fnp1, float64 dt) {
    const Domain *dom = &(conf->dom_);
    int s_nxmax  = dom->nxmax_[0];
    int s_nymax  = dom->nxmax_[1];
    int s_nvxmax = dom->nxmax_[2];

    Iterate_policy<3> policy3d({0, 0, 0}, {s_nxmax,s_nymax,s_nvxmax});
    Impl::for_each(policy3d, advect_1D_y_functor(conf, fn, fnp1, dt));
  };

  void advect_1D_vx(Config *conf, const RealView4D &fn, RealView4D &fnp1, Efield *ef, float64 dt) {
    const Domain *dom = &(conf->dom_);
    int s_nxmax  = dom->nxmax_[0];
    int s_nymax  = dom->nxmax_[1];
    int s_nvxmax = dom->nxmax_[2];

    Iterate_policy<3> policy3d({0, 0, 0}, {s_nxmax,s_nymax,s_nvxmax});
    Impl::for_each(policy3d, advect_1D_vx_functor(conf, fn, fnp1, ef, dt));
  };
  
  void advect_1D_vy(Config *conf, const RealView4D &fn, RealView4D &fnp1, Efield *ef, float64 dt) {
    const Domain *dom = &(conf->dom_);
    int s_nxmax  = dom->nxmax_[0];
    int s_nymax  = dom->nxmax_[1];
    int s_nvxmax = dom->nxmax_[2];

    Iterate_policy<3> policy3d({0, 0, 0}, {s_nxmax,s_nymax,s_nvxmax});
    Impl::for_each(policy3d, advect_1D_vy_functor(conf, fn, fnp1, ef, dt));
  };

  void print_fxvx(Config *conf, const RealView4D &fn, int iter) {
    Domain* dom = &(conf->dom_);
    char filename[128];
    printf("print_fxvx %d\n", iter);
    sprintf(filename, "fxvx%04d.out", iter);
    FILE* fileid = fopen(filename, "w");
    
    const int ivy = dom->nxmax_[3] / 2;
    const int iy = 0;
    
    for(int ivx = 0; ivx < dom->nxmax_[2]; ivx++)
    {
      for(int ix = 0; ix < dom->nxmax_[0]; ix++)
        fprintf(fileid, "%4d %4d %20.13le\n", ix, ivx, fn(ix, iy, ivx, ivy));
     
      fprintf(fileid, "\n");
    }
    
    fclose(fileid);
  };
};

#endif
