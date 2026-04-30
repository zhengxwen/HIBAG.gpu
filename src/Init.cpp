// ===============================================================
//
// HIBAG.gpu R package (GPU-based implementation for the HIBAG package)
// Copyright (C) 2026    Xiuwen Zheng (zhengx@u.washington.edu)
// All rights reserved.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.	 If not, see <http://www.gnu.org/licenses/>.

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

extern "C" {

extern SEXP gpu_init_proc(SEXP);
extern SEXP multigpu_init(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP multigpu_train(SEXP, SEXP);
extern SEXP ocl_set_verbose(SEXP);
extern SEXP ocl_init_dev_list();
extern SEXP ocl_dev_info(SEXP);
extern SEXP ocl_get_dev_param();
extern SEXP ocl_select_dev(SEXP);
extern SEXP ocl_release_dev();
extern SEXP ocl_set_local_size(SEXP, SEXP);
extern SEXP ocl_set_kl_attempt(SEXP, SEXP);
extern SEXP ocl_set_kl_clearmem(SEXP);
extern SEXP ocl_set_kl_build(SEXP, SEXP, SEXP);
extern SEXP ocl_set_kl_predict(SEXP, SEXP, SEXP, SEXP);
extern SEXP ocl_build_init(SEXP, SEXP, SEXP);
extern SEXP ocl_build_done();

static const R_CallMethodDef CallEntries[] = {
    { "gpu_init_proc",       (DL_FUNC)&gpu_init_proc,       1 },
    { "multigpu_init",       (DL_FUNC)&multigpu_init,       5 },
    { "multigpu_train",      (DL_FUNC)&multigpu_train,      2 },
    { "ocl_set_verbose",     (DL_FUNC)&ocl_set_verbose,     1 },
    { "ocl_init_dev_list",   (DL_FUNC)&ocl_init_dev_list,   0 },
    { "ocl_dev_info",        (DL_FUNC)&ocl_dev_info,        1 },
    { "ocl_get_dev_param",   (DL_FUNC)&ocl_get_dev_param,   0 },
    { "ocl_select_dev",      (DL_FUNC)&ocl_select_dev,      1 },
    { "ocl_release_dev",     (DL_FUNC)&ocl_release_dev,     0 },
    { "ocl_set_local_size",  (DL_FUNC)&ocl_set_local_size,  2 },
    { "ocl_set_kl_attempt",  (DL_FUNC)&ocl_set_kl_attempt,  2 },
    { "ocl_set_kl_clearmem", (DL_FUNC)&ocl_set_kl_clearmem, 1 },
    { "ocl_set_kl_build",    (DL_FUNC)&ocl_set_kl_build,    3 },
    { "ocl_set_kl_predict",  (DL_FUNC)&ocl_set_kl_predict,  4 },
    { "ocl_build_init",      (DL_FUNC)&ocl_build_init,      3 },
    { "ocl_build_done",      (DL_FUNC)&ocl_build_done,      0 },
    { NULL, NULL, 0 }
};

void R_init_HIBAG_gpu(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

} // extern "C"
