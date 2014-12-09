/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */

/** \file nbnxn_ocl_data_mgmt.cpp
 *  \brief OpenCL equivalent of nbnxn_cuda_data_mgmt.cu and cudautils.cu
 */

#include "config.h"

#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "gromacs/legacyheaders/tables.h"
#include "gromacs/legacyheaders/typedefs.h"
#include "gromacs/legacyheaders/types/enums.h"
#include "gromacs/mdlib/nb_verlet.h"
#include "gromacs/legacyheaders/types/interaction_const.h"
#include "gromacs/legacyheaders/types/force_flags.h"
#include "gromacs/mdlib/nbnxn_consts.h"
#include "gromacs/legacyheaders/gmx_detect_hardware.h"

#include "gromacs/mdlib/../gmxlib/ocl_tools/oclutils.h"
#include "gromacs/mdlib/nbnxn_ocl/nbnxn_ocl_types.h"
#include "gromacs/mdlib/nbnxn_gpu_data_mgmt.h"
#include "gromacs/legacyheaders/gpu_utils.h"

#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/timing/gpu_timing.h"


/* This is a heuristically determined parameter for the Fermi architecture for
 * the minimum size of ci lists by multiplying this constant with the # of
 * multiprocessors on the current device.
 */
static unsigned int gpu_min_ci_balanced_factor = 40;

/* We should actually be using md_print_warn in md_logging.c,
 * but we can't include mpi.h in CUDA code.
 */
static void md_print_warn(FILE       *fplog,
                          const char *fmt, ...)
{
    va_list ap;

    if (fplog != NULL)
    {
        /* We should only print to stderr on the master node,
         * in most cases fplog is only set on the master node, so this works.
         */
        va_start(ap, fmt);
        fprintf(stderr, "\n");
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
        va_end(ap);

        va_start(ap, fmt);
        fprintf(fplog, "\n");
        vfprintf(fplog, fmt, ap);
        fprintf(fplog, "\n");
        va_end(ap);
    }
}

/*!
 * If the pointers to the size variables are NULL no resetting happens.
 *  OpenCL equivalent of cu_free_buffered
 */
void ocl_free_buffered(cl_mem d_ptr, int *n, int *nalloc)
{
    cl_int cl_error;

    if (d_ptr)
    {
        cl_error = clReleaseMemObject(d_ptr);
        assert(cl_error == CL_SUCCESS);
        // TODO: handle errors
    }

    if (n)
    {
        *n = -1;
    }

    if (nalloc)
    {
        *nalloc = -1;
    }
}

/*!
 *  Reallocation of the memory pointed by d_ptr and copying of the data from
 *  the location pointed by h_src host-side pointer is done. Allocation is
 *  buffered and therefore freeing is only needed if the previously allocated
 *  space is not enough.
 *  The H2D copy is launched in command queue s and can be done synchronously or
 *  asynchronously (the default is the latter).
 *  If copy_event is not NULL, on return it will contain an event object
 *  identifying the H2D copy. The event can further be used to queue a wait
 *  for this operation or to query profiling information.
 *  OpenCL equivalent of cu_realloc_buffered.
 */
void ocl_realloc_buffered(cl_mem *d_dest, void *h_src,
                          size_t type_size,
                          int *curr_size, int *curr_alloc_size,
                          int req_size,
                          cl_context context,
                          cl_command_queue s,
                          bool bAsync = true,
                          cl_event *copy_event = NULL)
{
    cl_int cl_error;

    if (d_dest == NULL || req_size < 0)
    {
        return;
    }

    /* reallocate only if the data does not fit = allocation size is smaller
       than the current requested size */
    if (req_size > *curr_alloc_size)
    {
        /* only free if the array has already been initialized */
        if (*curr_alloc_size >= 0)
        {
            ocl_free_buffered(*d_dest, curr_size, curr_alloc_size);
        }

        *curr_alloc_size = over_alloc_large(req_size);

        *d_dest = clCreateBuffer(context, CL_MEM_READ_WRITE, *curr_alloc_size * type_size, NULL, &cl_error);
        assert(cl_error == CL_SUCCESS);
        // TODO: handle errors, check clCreateBuffer flags
    }

    /* size could have changed without actual reallocation */
    *curr_size = req_size;

    /* upload to device */
    if (h_src)
    {
        if (bAsync)
        {
            ocl_copy_H2D_async(*d_dest, h_src, 0, *curr_size * type_size, s, copy_event);
        }
        else
        {
            ocl_copy_H2D(*d_dest, h_src,  0, *curr_size * type_size, s);
        }
    }
}

/*! Tabulates the Ewald Coulomb force and initializes the size/scale
    and the table GPU array. If called with an already allocated table,
    it just re-uploads the table.

    OpenCL equivalent of init_ewald_coulomb_force_table from nbnxn_cuda_data_mgmt.cu
 */
static void init_ewald_coulomb_force_table(cl_nbparam_t             *nbp,
                                           const gmx_device_info_t *dev_info)
{
    float       *ftmp;
    cl_mem       coul_tab;
    int          tabsize;
    double       tabscale;

    cl_int       cl_error;

    tabsize     = GPU_EWALD_COULOMB_FORCE_TABLE_SIZE;
    /* Subtract 2 iso 1 to avoid access out of range due to rounding */
    tabscale    = (tabsize - 2) / sqrt(nbp->rcoulomb_sq);

    ocl_pmalloc((void**)&ftmp, tabsize*sizeof(*ftmp));

    table_spline3_fill_ewald_lr(ftmp, NULL, NULL, tabsize,
                                1/tabscale, nbp->ewald_beta, v_q_ewald_lr);

    /* If the table pointer == NULL the table is generated the first time =>
       the array pointer will be saved to nbparam and the texture is bound.
     */
    coul_tab = nbp->coulomb_tab_climg2d;
    if (coul_tab == NULL)
    {
        cl_image_format array_format;

        array_format.image_channel_data_type = CL_FLOAT;
        array_format.image_channel_order     = CL_R;

        /* Switched from using textures to using buffers */
        // TODO: decide which alternative is most efficient - textures or buffers.
        /*coul_tab = clCreateImage2D(dev_info->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            &array_format, tabsize, 1, 0, ftmp, &cl_error);*/
        coul_tab = clCreateBuffer(dev_info->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tabsize*sizeof(cl_float), ftmp, &cl_error);
        assert(cl_error == CL_SUCCESS);
        // TODO: handle errors, check clCreateBuffer flags

        nbp->coulomb_tab_climg2d  = coul_tab;
        nbp->coulomb_tab_size     = tabsize;
        nbp->coulomb_tab_scale    = tabscale;
    }

    ocl_pfree(ftmp);
}


/*! Initializes the atomdata structure first time, it only gets filled at
    pair-search.
    OpenCL equivalent of init_atomdata_first from nbnxn_cuda_data_mgmt.cu
 */
static void init_atomdata_first(cl_atomdata_t *ad, int ntypes, gmx_device_info_t *dev_info)
{
    cl_int cl_error;

    ad->ntypes  = ntypes;

    ad->shift_vec = clCreateBuffer(dev_info->context, CL_MEM_READ_WRITE, SHIFTS * sizeof(rvec), NULL, &cl_error);
    assert(cl_error == CL_SUCCESS);
    ad->bShiftVecUploaded = false;
    // TODO: handle errors, check clCreateBuffer flags

    ad->fshift = clCreateBuffer(dev_info->context, CL_MEM_READ_WRITE, SHIFTS * sizeof(rvec), NULL, &cl_error);
    assert(cl_error == CL_SUCCESS);
    // TODO: handle errors, check clCreateBuffer flags

    ad->e_lj = clCreateBuffer(dev_info->context, CL_MEM_READ_WRITE, sizeof(float), NULL, &cl_error);
    assert(cl_error == CL_SUCCESS);
    // TODO: handle errors, check clCreateBuffer flags

    ad->e_el = clCreateBuffer(dev_info->context, CL_MEM_READ_WRITE, sizeof(float), NULL, &cl_error);
    assert(cl_error == CL_SUCCESS);
    // TODO: handle errors, check clCreateBuffer flags

    /* initialize to NULL pointers to data that is not allocated here and will
       need reallocation in nbnxn_gpu_init_atomdata */
    ad->xq = NULL;
    ad->f  = NULL;

    /* size -1 indicates that the respective array hasn't been initialized yet */
    ad->natoms = -1;
    ad->nalloc = -1;
}

/*! Selects the Ewald kernel type, analytical or tabulated, single or twin cut-off.
    OpenCL equivalent of pick_ewald_kernel_type from nbnxn_cuda_data_mgmt.cu
 */
static int pick_ewald_kernel_type(bool bTwinCut)
{
    bool bUseAnalyticalEwald, bForceAnalyticalEwald, bForceTabulatedEwald;
    int  kernel_type;

    /* Benchmarking/development environment variables to force the use of
       analytical or tabulated Ewald kernel. */
    bForceAnalyticalEwald = (getenv("GMX_OCL_NB_ANA_EWALD") != NULL);
    bForceTabulatedEwald  = (getenv("GMX_OCL_NB_TAB_EWALD") != NULL);

    if (bForceAnalyticalEwald && bForceTabulatedEwald)
    {
        gmx_incons("Both analytical and tabulated Ewald OpenCL non-bonded kernels "
                   "requested through environment variables.");
    }

    /* CUDA: By default, on SM 3.0 and later use analytical Ewald, on earlier tabulated. */
    /* OpenCL: By default, use analytical Ewald, on earlier tabulated. */
    // TODO: decide if dev_info parameter should be added to recognize NVIDIA CC>=3.0 devices.
    //if ((dev_info->prop.major >= 3 || bForceAnalyticalEwald) && !bForceTabulatedEwald)
    if ((1                         || bForceAnalyticalEwald) && !bForceTabulatedEwald)
    {
        bUseAnalyticalEwald = true;

        if (debug)
        {
            fprintf(debug, "Using analytical Ewald OpenCL kernels\n");
        }
    }
    else
    {
        bUseAnalyticalEwald = false;

        if (debug)
        {
            fprintf(debug, "Using tabulated Ewald OpenCL kernels\n");
        }
    }

    /* Use twin cut-off kernels if requested by bTwinCut or the env. var.
       forces it (use it for debugging/benchmarking only). */
    if (!bTwinCut && (getenv("GMX_OCL_NB_EWALD_TWINCUT") == NULL))
    {
        kernel_type = bUseAnalyticalEwald ? eelOclEWALD_ANA : eelOclEWALD_TAB;
    }
    else
    {
        kernel_type = bUseAnalyticalEwald ? eelOclEWALD_ANA_TWIN : eelOclEWALD_TAB_TWIN;
    }

    return kernel_type;
}

/*! Copies all parameters related to the cut-off from ic to nbp
    OpenCL equivalent of set_cutoff_parameters from nbnxn_cuda_data_mgmt.cu
 */
static void set_cutoff_parameters(cl_nbparam_t              *nbp,
                                  const interaction_const_t *ic)
{
    nbp->ewald_beta       = ic->ewaldcoeff_q;
    nbp->sh_ewald         = ic->sh_ewald;
    nbp->epsfac           = ic->epsfac;
    nbp->two_k_rf         = 2.0 * ic->k_rf;
    nbp->c_rf             = ic->c_rf;
    nbp->rvdw_sq          = ic->rvdw * ic->rvdw;
    nbp->rcoulomb_sq      = ic->rcoulomb * ic->rcoulomb;
    nbp->rlist_sq         = ic->rlist * ic->rlist;

    nbp->sh_lj_ewald      = ic->sh_lj_ewald;
    nbp->ewaldcoeff_lj    = ic->ewaldcoeff_lj;

    nbp->rvdw_switch      = ic->rvdw_switch;
    nbp->dispersion_shift = ic->dispersion_shift;
    nbp->repulsion_shift  = ic->repulsion_shift;
    nbp->vdw_switch       = ic->vdw_switch;
}
/*! Determines the families of electrostatics and Vdw OpenCL kernels.
 */
void nbnxn_ocl_convert_gmx_to_gpu_flavors(
        const int gmx_eeltype,
        const int gmx_vdwtype,
        const int gmx_vdw_modifier,
        const int gmx_ljpme_comb_rule,
        int      *gpu_eeltype,
        int      *gpu_vdwtype)
{
    if (gmx_vdwtype == evdwCUT)
    {
        switch (gmx_vdw_modifier)
        {
            case eintmodNONE:
            case eintmodPOTSHIFT:
                *gpu_vdwtype = evdwOclCUT;
                break;
            case eintmodFORCESWITCH:
                *gpu_vdwtype = evdwOclFSWITCH;
                break;
            case eintmodPOTSWITCH:
                *gpu_vdwtype = evdwOclPSWITCH;
                break;
            default:
                gmx_incons("The requested VdW interaction modifier is not implemented in the GPU accelerated kernels!");
                break;
        }
    }
    else if (gmx_vdwtype == evdwPME)
    {
        if (gmx_ljpme_comb_rule == ljcrGEOM)
        {
            *gpu_vdwtype = evdwOclEWALDGEOM;
        }
        else
        {
            *gpu_vdwtype = evdwOclEWALDLB;
        }
    }
    else
    {
        gmx_incons("The requested VdW type is not implemented in the GPU accelerated kernels!");
    }

    if (gmx_eeltype == eelCUT)
    {
        *gpu_eeltype = eelOclCUT;
    }
    else if (EEL_RF(gmx_eeltype))
    {
        *gpu_eeltype = eelOclRF;
    }
    else if ((EEL_PME(gmx_eeltype) || gmx_eeltype == eelEWALD))
    {
        /* Initially rcoulomb == rvdw, so it's surely not twin cut-off. */
        *gpu_eeltype = pick_ewald_kernel_type(false);
    }
    else
    {
        /* Shouldn't happen, as this is checked when choosing Verlet-scheme */
        gmx_incons("The requested electrostatics type is not implemented in the GPU accelerated kernels!");
    }
}

/*! Initializes the nonbonded parameter data structure.
    OpenCL equivalent of init_nbparam from nbnxn_cuda_data_mgmt.cu
 */
static void init_nbparam(cl_nbparam_t              *nbp,
                         const interaction_const_t *ic,
                         const nbnxn_atomdata_t    *nbat,
                         const gmx_device_info_t   *dev_info)
{
    int         ntypes, nnbfp, nnbfp_comb;
    cl_int      cl_error;


    ntypes = nbat->ntype;

    set_cutoff_parameters(nbp, ic);

    nbnxn_ocl_convert_gmx_to_gpu_flavors(
            ic->eeltype,
            ic->vdwtype,
            ic->vdw_modifier,
            ic->ljpme_comb_rule,
            &(nbp->eeltype),
            &(nbp->vdwtype));

    if (ic->vdwtype == evdwPME)
    {
        if (ic->ljpme_comb_rule == ljcrGEOM)
        {
            assert(nbat->comb_rule == ljcrGEOM);
        }
        else
        {
            assert(nbat->comb_rule == ljcrLB);
        }
    }
    /* generate table for PME */
    nbp->coulomb_tab_climg2d = NULL;
    if (nbp->eeltype == eelOclEWALD_TAB || nbp->eeltype == eelOclEWALD_TAB_TWIN)
    {
        init_ewald_coulomb_force_table(nbp, dev_info);
    }
    else
    // TODO: improvement needed.
    // The image2d is created here even if eeltype is not eelCuEWALD_TAB or eelCuEWALD_TAB_TWIN because the OpenCL kernels
    // don't accept NULL values for image2D parameters.
    {
        cl_image_format array_format;

        array_format.image_channel_data_type = CL_FLOAT;
        array_format.image_channel_order     = CL_R;

        /* Switched from using textures to using buffers */
        // TODO: decide which alternative is most efficient - textures or buffers.
        /*nbp->coulomb_tab_climg2d = clCreateImage2D(dev_info->context, CL_MEM_READ_WRITE,
            &array_format, 1, 1, 0, NULL, &cl_error);*/
        nbp->coulomb_tab_climg2d = clCreateBuffer(dev_info->context, CL_MEM_READ_ONLY, sizeof(cl_float), NULL, &cl_error);
        // TODO: handle errors
    }

    nnbfp      = 2*ntypes*ntypes;
    nnbfp_comb = 2*ntypes;

    {
        cl_image_format array_format;

        array_format.image_channel_data_type = CL_FLOAT;
        array_format.image_channel_order     = CL_R;

        /* Switched from using textures to using buffers */
        // TODO: decide which alternative is most efficient - textures or buffers.

        /*nbp->nbfp_climg2d = clCreateImage2D(dev_info->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &array_format, nnbfp, 1, 0, nbat->nbfp, &cl_error);*/
        nbp->nbfp_climg2d = clCreateBuffer(dev_info->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nnbfp*sizeof(cl_float), nbat->nbfp, &cl_error);
        assert(cl_error == CL_SUCCESS);
        // TODO: handle errors

        if (ic->vdwtype == evdwPME)
        {
            /* Switched from using textures to using buffers */
            // TODO: decide which alternative is most efficient - textures or buffers.
            /*  nbp->nbfp_comb_climg2d = clCreateImage2D(dev_info->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                &array_format, nnbfp_comb, 1, 0, nbat->nbfp_comb, &cl_error);*/
            nbp->nbfp_comb_climg2d = clCreateBuffer(dev_info->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nnbfp_comb*sizeof(cl_float), nbat->nbfp_comb, &cl_error);


            assert(cl_error == CL_SUCCESS);
            // TODO: handle errors
        }
        else
        {
            // TODO: improvement needed.
            // The image2d is created here even if vdwtype is not evdwPME because the OpenCL kernels
            // don't accept NULL values for image2D parameters.
            /* Switched from using textures to using buffers */
            // TODO: decide which alternative is most efficient - textures or buffers.
            /* nbp->nbfp_comb_climg2d = clCreateImage2D(dev_info->context, CL_MEM_READ_WRITE,
                &array_format, 1, 1, 0, NULL, &cl_error);*/
            nbp->nbfp_comb_climg2d = clCreateBuffer(dev_info->context, CL_MEM_READ_ONLY, sizeof(cl_float), NULL, &cl_error);


            assert(cl_error == CL_SUCCESS);
            // TODO: handle errors
        }
    }
}

/*! Re-generate the GPU Ewald force table, resets rlist, and update the
 *  electrostatic type switching to twin cut-off (or back) if needed.
 *  OpenCL equivalent of nbnxn_cuda_pme_loadbal_update_param
 */
void nbnxn_gpu_pme_loadbal_update_param(const nonbonded_verlet_t    *nbv,
                                        const interaction_const_t   *ic)
{
    if (!nbv || nbv->grp[0].kernel_type != nbnxnk8x8x8_CUDA)
    {
        return;
    }
    gmx_nbnxn_ocl_t *nb  = nbv->gpu_nbv;
    cl_nbparam_t       *nbp = nb->nbparam;

    set_cutoff_parameters(nbp, ic);

    nbp->eeltype = pick_ewald_kernel_type(ic->rcoulomb != ic->rvdw);

    init_ewald_coulomb_force_table(nb->nbparam, nb->dev_info);
}

/*! Initializes the pair list data structure.
 *  OpenCL equivalent of init_plist from nbnxn_cuda_data_mgmt.cu
 */
static void init_plist(cl_plist_t *pl)
{
    /* initialize to NULL pointers to data that is not allocated here and will
       need reallocation in nbnxn_gpu_init_pairlist */
    pl->sci     = NULL;
    pl->cj4     = NULL;
    pl->excl    = NULL;

    /* size -1 indicates that the respective array hasn't been initialized yet */
    pl->na_c        = -1;
    pl->nsci        = -1;
    pl->sci_nalloc  = -1;
    pl->ncj4        = -1;
    pl->cj4_nalloc  = -1;
    pl->nexcl       = -1;
    pl->excl_nalloc = -1;
    pl->bDoPrune    = false;
}

/*! Initializes the timer data structure.
    OpenCL equivalent of init_timers from nbnxn_cuda_data_mgmt.cu
 */
static void init_timers(cl_timers_t *t, bool bUseTwoStreams)
{
    /* Nothing to initialize for OpenCL */
}

/*! Initializes the timings data structure.
    OpenCL equivalent of init_timings from nbnxn_cuda_data_mgmt.cu
 */
static void init_timings(gmx_wallclock_gpu_t *t)
{
    int i, j;

    t->nb_h2d_t = 0.0;
    t->nb_d2h_t = 0.0;
    t->nb_c     = 0;
    t->pl_h2d_t = 0.0;
    t->pl_h2d_c = 0;
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            t->ktime[i][j].t = 0.0;
            t->ktime[i][j].c = 0;
        }
    }
}

/*! Initializes the OpenCL kernel pointers of the nbnxn_ocl_ptr_t input data structure. */
void nbnxn_init_kernels(gmx_nbnxn_ocl_t *nb)
{
    cl_int cl_error;

    /* Init to 0 main kernel arrays */
    /* They will be later on initialized in select_nbnxn_kernel */
    memset(nb->kernel_ener_noprune_ptr, 0, sizeof(nb->kernel_ener_noprune_ptr));
    memset(nb->kernel_ener_prune_ptr, 0, sizeof(nb->kernel_ener_prune_ptr));
    memset(nb->kernel_noener_noprune_ptr, 0, sizeof(nb->kernel_noener_noprune_ptr));
    memset(nb->kernel_noener_prune_ptr, 0, sizeof(nb->kernel_noener_prune_ptr));

    /* Init auxiliary kernels */
    nb->kernel_memset_f = clCreateKernel(nb->dev_info->program, "memset_f", &cl_error);
    assert(cl_error == CL_SUCCESS);

    nb->kernel_memset_f2 = clCreateKernel(nb->dev_info->program, "memset_f2", &cl_error);
    assert(cl_error == CL_SUCCESS);

    nb->kernel_memset_f3 = clCreateKernel(nb->dev_info->program, "memset_f3", &cl_error);
    assert(cl_error == CL_SUCCESS);

    nb->kernel_zero_e_fshift = clCreateKernel(nb->dev_info->program, "zero_e_fshift", &cl_error);
    assert(cl_error == CL_SUCCESS);
}

/*! Initializes the input nbnxn_ocl_ptr_t data structure.
    OpenCL equivalent of nbnxn_cuda_init
 */
void nbnxn_gpu_init(FILE                 *fplog,
                    gmx_nbnxn_ocl_t **p_nb,
                    const gmx_gpu_info_t *gpu_info,
                    const gmx_gpu_opt_t  *gpu_opt,
                    int                   my_gpu_index,
                    gmx_bool              bLocalAndNonlocal)
{
    gmx_nbnxn_ocl_t *nb;
    cl_int                      cl_error;
    char                        sbuf[STRLEN];
    bool                        bStreamSync, bNoStreamSync, bTMPIAtomics, bX86, bOldDriver;
    cl_command_queue_properties queue_properties;

    assert(gpu_info);

    if (p_nb == NULL)
    {
        return;
    }

    snew(nb, 1);
    snew(nb->atdat, 1);
    snew(nb->nbparam, 1);
    snew(nb->plist[eintLocal], 1);
    if (bLocalAndNonlocal)
    {
        snew(nb->plist[eintNonlocal], 1);
    }

    nb->bUseTwoStreams = bLocalAndNonlocal;

    snew(nb->timers, 1);
    snew(nb->timings, 1);

    /* set device info, just point it to the right GPU among the detected ones */
    nb->dev_info = gpu_info->gpu_dev + gpu_opt->dev_use[my_gpu_index];

    /* init the kernels */
    nbnxn_init_kernels(nb);

    /* init to NULL the debug buffer */
    nb->debug_buffer = NULL;

    /* init nbst */
    ocl_pmalloc((void**)&nb->nbst.e_lj, sizeof(*nb->nbst.e_lj));
    ocl_pmalloc((void**)&nb->nbst.e_el, sizeof(*nb->nbst.e_el));

    // TODO: review fshift data type and how its size is computed
    ocl_pmalloc((void**)&nb->nbst.fshift, 3 * SHIFTS * sizeof(*nb->nbst.fshift));

    init_plist(nb->plist[eintLocal]);

    // TODO: Update the code below for OpenCL and for NVIDIA GPUs.
    // For now, bUseStreamSync will always be true.
    nb->bUseStreamSync = true;

    /* On GPUs with ECC enabled, cudaStreamSynchronize shows a large overhead
     * (which increases with shorter time/step) caused by a known CUDA driver bug.
     * To work around the issue we'll use an (admittedly fragile) memory polling
     * waiting to preserve performance. This requires support for atomic
     * operations and only works on x86/x86_64.
     * With polling wait event-timing also needs to be disabled.
     *
     * The overhead is greatly reduced in API v5.0 drivers and the improvement
     * is independent of runtime version. Hence, with API v5.0 drivers and later
     * we won't switch to polling.
     *
     * NOTE: Unfortunately, this is known to fail when GPUs are shared by (t)MPI,
     * ranks so we will also disable it in that case.
     */

//////    bStreamSync    = getenv("GMX_CUDA_STREAMSYNC") != NULL;
//////    bNoStreamSync  = getenv("GMX_NO_CUDA_STREAMSYNC") != NULL;
//////
//////#ifdef TMPI_ATOMICS
//////    bTMPIAtomics = true;
//////#else
//////    bTMPIAtomics = false;
//////#endif
//////
//////#ifdef GMX_TARGET_X86
//////    bX86 = true;
//////#else
//////    bX86 = false;
//////#endif
//////
//////    if (bStreamSync && bNoStreamSync)
//////    {
//////        gmx_fatal(FARGS, "Conflicting environment variables: both GMX_CUDA_STREAMSYNC and GMX_NO_CUDA_STREAMSYNC defined");
//////    }
//////
//////
//////    stat = cudaDriverGetVersion(&cuda_drv_ver);
//////    CU_RET_ERR(stat, "cudaDriverGetVersion failed");
//////
//////    bOldDriver = (cuda_drv_ver < 5000);
//////
//////    if ((nb->dev_info->prop.ECCEnabled == 1) && bOldDriver)
//////    {
//////        /* Polling wait should be used instead of cudaStreamSynchronize only if:
//////         *   - ECC is ON & driver is old (checked above),
//////         *   - we're on x86/x86_64,
//////         *   - atomics are available, and
//////         *   - GPUs are not being shared.
//////         */
//////        bool bShouldUsePollSync = (bX86 && bTMPIAtomics &&
//////                                   (gmx_count_gpu_dev_shared(gpu_opt) < 1));
//////
//////        if (bStreamSync)
//////        {
//////            nb->bUseStreamSync = true;
//////
//////            /* only warn if polling should be used */
//////            if (bShouldUsePollSync)
//////            {
//////                md_print_warn(fplog,
//////                              "NOTE: Using a GPU with ECC enabled and CUDA driver API version <5.0, but\n"
//////                              "      cudaStreamSynchronize waiting is forced by the GMX_CUDA_STREAMSYNC env. var.\n");
//////            }
//////        }
//////        else
//////        {
//////            nb->bUseStreamSync = !bShouldUsePollSync;
//////
//////            if (bShouldUsePollSync)
//////            {
//////                md_print_warn(fplog,
//////                              "NOTE: Using a GPU with ECC enabled and CUDA driver API version <5.0, known to\n"
//////                              "      cause performance loss. Switching to the alternative polling GPU wait.\n"
//////                              "      If you encounter issues, switch back to standard GPU waiting by setting\n"
//////                              "      the GMX_CUDA_STREAMSYNC environment variable.\n");
//////            }
//////            else
//////            {
//////                /* Tell the user that the ECC+old driver combination can be bad */
//////                sprintf(sbuf,
//////                        "NOTE: Using a GPU with ECC enabled and CUDA driver API version <5.0.\n"
//////                        "      A known bug in this driver version can cause performance loss.\n"
//////                        "      However, the polling wait workaround can not be used because\n%s\n"
//////                        "      Consider updating the driver or turning ECC off.",
//////                        (bX86 && bTMPIAtomics) ?
//////                        "      GPU(s) are being oversubscribed." :
//////                        "      atomic operations are not supported by the platform/CPU+compiler.");
//////                md_print_warn(fplog, sbuf);
//////            }
//////        }
//////    }
//////    else
//////    {
//////        if (bNoStreamSync)
//////        {
//////            nb->bUseStreamSync = false;
//////
//////            md_print_warn(fplog,
//////                          "NOTE: Polling wait for GPU synchronization requested by GMX_NO_CUDA_STREAMSYNC\n");
//////        }
//////        else
//////        {
//////            /* no/off ECC, cudaStreamSynchronize not turned off by env. var. */
//////            nb->bUseStreamSync = true;
//////        }
//////    }



    /* OpenCL timing disabled as event timers don't work:
       - with multiple streams = domain-decomposition;
       - with the polling waiting hack (without cudaStreamSynchronize);
       - when turned off by GMX_DISABLE_OCL_TIMING.
     */
    nb->bDoTime = (!nb->bUseTwoStreams && nb->bUseStreamSync &&
                   (getenv("GMX_DISABLE_OCL_TIMING") == NULL));

    /* Create queues only after bDoTime has been initialized */
    if (nb->bDoTime)
    {
        queue_properties = CL_QUEUE_PROFILING_ENABLE;
    }
    else
    {
        queue_properties = 0;
    }

    /* local/non-local GPU streams */
    nb->stream[eintLocal] = clCreateCommandQueue(nb->dev_info->context, nb->dev_info->ocl_gpu_id.ocl_device_id, queue_properties, &cl_error);
    assert(cl_error == CL_SUCCESS);
    // TODO: check for errors

    if (nb->bUseTwoStreams)
    {
        init_plist(nb->plist[eintNonlocal]);

        nb->stream[eintNonlocal] = clCreateCommandQueue(nb->dev_info->context, nb->dev_info->ocl_gpu_id.ocl_device_id, queue_properties, &cl_error);
        assert(cl_error == CL_SUCCESS);
        // TODO: check for errors
    }

    if (nb->bDoTime)
    {
        init_timers(nb->timers, nb->bUseTwoStreams);
        init_timings(nb->timings);
    }

    // TODO: check if it's worth implementing for NVIDIA GPUs
    ///////////* set the kernel type for the current GPU */
    ///////////* pick L1 cache configuration */
    //////////nbnxn_gpu_set_cacheconfig(nb->dev_info);

    *p_nb = nb;

    if (debug)
    {
        fprintf(debug, "Initialized OpenCL data structures.\n");
    }
}

/*! Clears nonbonded shift force output array and energy outputs on the GPU.
    OpenCL equivalent of nbnxn_cuda_clear_e_fshift
 */
static void
nbnxn_ocl_clear_e_fshift(gmx_nbnxn_ocl_t *nb)
{

    cl_int               cl_error = CL_SUCCESS;
    cl_atomdata_t *      adat     = nb->atdat;
    cl_command_queue     ls       = nb->stream[eintLocal];
    gmx_device_info_t *dev_info = nb->dev_info;

    size_t               dim_block[3] = {1, 1, 1};
    size_t               dim_grid[3]  = {1, 1, 1};
    cl_int               shifts       = SHIFTS*3;

    cl_int               arg_no;

    cl_kernel            zero_e_fshift = nb->kernel_zero_e_fshift;

    dim_block[0] = 64;
    dim_grid[0]  = ((shifts/64)*64) + ((shifts%64) ? 64 : 0);

    arg_no    = 0;
    cl_error  = clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_mem), &(adat->fshift));
    cl_error |= clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_mem), &(adat->e_lj));
    cl_error |= clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_mem), &(adat->e_el));
    cl_error |= clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_uint), &shifts);
    assert(cl_error == CL_SUCCESS);

    cl_error = clEnqueueNDRangeKernel(ls, zero_e_fshift, 3, NULL, dim_grid, dim_block, 0, NULL, NULL);
    assert(cl_error == CL_SUCCESS);

}

/*! Clears the first natoms_clear elements of the GPU nonbonded force output array.
    OpenCL equivalent of nbnxn_cuda_clear_f
 */
static void nbnxn_ocl_clear_f(gmx_nbnxn_ocl_t *nb, int natoms_clear)
{

    cl_int               cl_error = CL_SUCCESS;
    cl_atomdata_t *      adat     = nb->atdat;
    cl_command_queue     ls       = nb->stream[eintLocal];
    gmx_device_info_t *dev_info = nb->dev_info;
    cl_float             value    = 0.0f;

    size_t               dim_block[3] = {1, 1, 1};
    size_t               dim_grid[3]  = {1, 1, 1};

    cl_int               arg_no;

    cl_kernel            memset_f = nb->kernel_memset_f;

    cl_uint              natoms_flat = natoms_clear * (sizeof(rvec)/sizeof(real));

    dim_block[0] = 64;
    dim_grid[0]  = ((natoms_flat/dim_block[0])*dim_block[0]) + ((natoms_flat%dim_block[0]) ? dim_block[0] : 0);

    arg_no    = 0;
    cl_error  = clSetKernelArg(memset_f, arg_no++, sizeof(cl_mem), &(adat->f));
    cl_error  = clSetKernelArg(memset_f, arg_no++, sizeof(cl_float), &value);
    cl_error |= clSetKernelArg(memset_f, arg_no++, sizeof(cl_uint), &natoms_flat);
    assert(cl_error == CL_SUCCESS);

    cl_error = clEnqueueNDRangeKernel(ls, memset_f, 3, NULL, dim_grid, dim_block, 0, NULL, NULL);
    assert(cl_error == CL_SUCCESS);
}

/*! OpenCL equivalent of nbnxn_cuda_clear_outputs */
void
nbnxn_gpu_clear_outputs(gmx_nbnxn_ocl_t *nb,
                        int                flags)
{
    nbnxn_ocl_clear_f(nb, nb->atdat->natoms);
    /* clear shift force array and energies if the outputs were
       used in the current step */
    if (flags & GMX_FORCE_VIRIAL)
    {
        nbnxn_ocl_clear_e_fshift(nb);
    }
}

/*! OpenCL equivalent of nbnxn_cuda_init_const */
void nbnxn_gpu_init_const(gmx_nbnxn_ocl_t *nb,
                          const interaction_const_t      *ic,
                          const nonbonded_verlet_group_t *nbv_group)
{
    init_atomdata_first(nb->atdat, nbv_group[0].nbat->ntype, nb->dev_info);

    init_nbparam(nb->nbparam, ic, nbv_group[0].nbat, nb->dev_info);

    /* clear energy and shift force outputs */
    nbnxn_ocl_clear_e_fshift(nb);
}

/*! OpenCL equivalent of nbnxn_cuda_init_pairlist */
void nbnxn_gpu_init_pairlist(gmx_nbnxn_ocl_t *nb,
                             const nbnxn_pairlist_t *h_plist,
                             int                     iloc)
{
    char             sbuf[STRLEN];
    bool             bDoTime    = nb->bDoTime;
    cl_command_queue stream     = nb->stream[iloc];
    cl_plist_t      *d_plist    = nb->plist[iloc];

    if (d_plist->na_c < 0)
    {
        d_plist->na_c = h_plist->na_ci;
    }
    else
    {
        if (d_plist->na_c != h_plist->na_ci)
        {
            sprintf(sbuf, "In cu_init_plist: the #atoms per cell has changed (from %d to %d)",
                    d_plist->na_c, h_plist->na_ci);
            gmx_incons(sbuf);
        }
    }

    ocl_realloc_buffered(&d_plist->sci, h_plist->sci, sizeof(nbnxn_sci_t),
                         &d_plist->nsci, &d_plist->sci_nalloc,
                         h_plist->nsci,
                         nb->dev_info->context,
                         stream, true, &(nb->timers->pl_h2d_sci[iloc]));

    ocl_realloc_buffered(&d_plist->cj4, h_plist->cj4, sizeof(nbnxn_cj4_t),
                         &d_plist->ncj4, &d_plist->cj4_nalloc,
                         h_plist->ncj4,
                         nb->dev_info->context,
                         stream, true, &(nb->timers->pl_h2d_cj4[iloc]));

    ocl_realloc_buffered(&d_plist->excl, h_plist->excl, sizeof(nbnxn_excl_t),
                         &d_plist->nexcl, &d_plist->excl_nalloc,
                         h_plist->nexcl,
                         nb->dev_info->context,
                         stream, true, &(nb->timers->pl_h2d_excl[iloc]));

    /* need to prune the pair list during the next step */
    d_plist->bDoPrune = true;
}

/*! OpenCL equivalent of nbnxn_cuda_upload_shiftvec */
void nbnxn_gpu_upload_shiftvec(gmx_nbnxn_ocl_t *nb,
                               const nbnxn_atomdata_t *nbatom)
{
    cl_atomdata_t   *adat  = nb->atdat;
    cl_command_queue ls    = nb->stream[eintLocal];

    /* only if we have a dynamic box */
    if (nbatom->bDynamicBox || !adat->bShiftVecUploaded)
    {
        ocl_copy_H2D_async(adat->shift_vec, nbatom->shift_vec, 0,
                           SHIFTS * sizeof(rvec), ls, NULL);
        adat->bShiftVecUploaded = true;
    }
}

/*! Initialize atomdata */
void nbnxn_gpu_init_atomdata(gmx_nbnxn_ocl_t *nb,
                             const struct nbnxn_atomdata_t *nbat)
{
    cl_int           cl_error;
    int              nalloc, natoms;
    bool             realloced;
    bool             bDoTime = nb->bDoTime;
    cl_timers_t     *timers  = nb->timers;
    cl_atomdata_t   *d_atdat = nb->atdat;
    cl_command_queue ls      = nb->stream[eintLocal];

    natoms    = nbat->natoms;
    realloced = false;

    /* need to reallocate if we have to copy more atoms than the amount of space
       available and only allocate if we haven't initialized yet, i.e d_atdat->natoms == -1 */
    if (natoms > d_atdat->nalloc)
    {
        nalloc = over_alloc_small(natoms);

        /* free up first if the arrays have already been initialized */
        if (d_atdat->nalloc != -1)
        {
            ocl_free_buffered(d_atdat->f, &d_atdat->natoms, &d_atdat->nalloc);
            ocl_free_buffered(d_atdat->xq, NULL, NULL);
            ocl_free_buffered(d_atdat->atom_types, NULL, NULL);
        }

        d_atdat->f = clCreateBuffer(nb->dev_info->context, CL_MEM_READ_WRITE, nalloc * sizeof(rvec), NULL, &cl_error);
        assert(CL_SUCCESS == cl_error);
        // TODO: handle errors, check clCreateBuffer flags

        d_atdat->xq = clCreateBuffer(nb->dev_info->context, CL_MEM_READ_WRITE, nalloc * sizeof(cl_float4), NULL, &cl_error);
        assert(CL_SUCCESS == cl_error);
        // TODO: handle errors, check clCreateBuffer flags

        d_atdat->atom_types = clCreateBuffer(nb->dev_info->context, CL_MEM_READ_WRITE, nalloc * sizeof(int), NULL, &cl_error);
        assert(CL_SUCCESS == cl_error);
        // TODO: handle errors, check clCreateBuffer flags

        d_atdat->nalloc = nalloc;
        realloced       = true;
    }

    d_atdat->natoms       = natoms;
    d_atdat->natoms_local = nbat->natoms_local;

    /* need to clear GPU f output if realloc happened */
    if (realloced)
    {
        nbnxn_ocl_clear_f(nb, nalloc);
    }

    ocl_copy_H2D_async(d_atdat->atom_types, nbat->type, 0,
                       natoms*sizeof(int), ls, bDoTime ? &(timers->atdat) : NULL);
}

/*! Releases an OpenCL kernel pointer */
void free_kernel(cl_kernel *kernel_ptr)
{
    cl_int cl_error;

    assert(NULL != kernel_ptr);

    if (*kernel_ptr)
    {
        cl_error = clReleaseKernel(*kernel_ptr);
        assert(cl_error == CL_SUCCESS);

        *kernel_ptr = NULL;
    }
}

/*! Releases a list of OpenCL kernel pointers */
void free_kernels(cl_kernel *kernels, int count)
{
    int i;

    for (i = 0; i < count; i++)
    {
        free_kernel(kernels + i);
    }
}

/*! Releases the input OpenCL buffer */
void free_ocl_buffer(cl_mem *buffer)
{
    cl_int cl_error;

    assert(NULL != buffer);

    if (*buffer)
    {
        cl_error = clReleaseMemObject(*buffer);
        assert(CL_SUCCESS == cl_error);
        *buffer = NULL;
    }
}

/*! OpenCL equivalent of nbnxn_cuda_free */
void nbnxn_gpu_free(gmx_nbnxn_ocl_t *nb)
{
    // TODO: Implement this functions for OpenCL
    cl_int cl_error;
    int    kernel_count;

    /* Free kernels */
    kernel_count = sizeof(nb->kernel_ener_noprune_ptr) / sizeof(nb->kernel_ener_noprune_ptr[0][0]);
    free_kernels((cl_kernel*)nb->kernel_ener_noprune_ptr, kernel_count);

    kernel_count = sizeof(nb->kernel_ener_prune_ptr) / sizeof(nb->kernel_ener_prune_ptr[0][0]);
    free_kernels((cl_kernel*)nb->kernel_ener_prune_ptr, kernel_count);

    kernel_count = sizeof(nb->kernel_noener_noprune_ptr) / sizeof(nb->kernel_noener_noprune_ptr[0][0]);
    free_kernels((cl_kernel*)nb->kernel_noener_noprune_ptr, kernel_count);

    kernel_count = sizeof(nb->kernel_noener_prune_ptr) / sizeof(nb->kernel_noener_prune_ptr[0][0]);
    free_kernels((cl_kernel*)nb->kernel_noener_prune_ptr, kernel_count);

    free_kernel(&(nb->kernel_memset_f));
    free_kernel(&(nb->kernel_memset_f2));
    free_kernel(&(nb->kernel_memset_f3));
    free_kernel(&(nb->kernel_zero_e_fshift));

    /* Free atdat */
    free_ocl_buffer(&(nb->atdat->xq));
    free_ocl_buffer(&(nb->atdat->f));
    free_ocl_buffer(&(nb->atdat->e_lj));
    free_ocl_buffer(&(nb->atdat->e_el));
    free_ocl_buffer(&(nb->atdat->fshift));
    free_ocl_buffer(&(nb->atdat->atom_types));
    free_ocl_buffer(&(nb->atdat->shift_vec));
    sfree(nb->atdat);

    /* Free nbparam */
    free_ocl_buffer(&(nb->nbparam->nbfp_climg2d));
    free_ocl_buffer(&(nb->nbparam->nbfp_comb_climg2d));
    free_ocl_buffer(&(nb->nbparam->coulomb_tab_climg2d));
    sfree(nb->nbparam);

    /* Free plist */
    free_ocl_buffer(&(nb->plist[eintLocal]->sci));
    free_ocl_buffer(&(nb->plist[eintLocal]->cj4));
    free_ocl_buffer(&(nb->plist[eintLocal]->excl));
    sfree(nb->plist[eintLocal]);
    if (nb->bUseTwoStreams)
    {
        free_ocl_buffer(&(nb->plist[eintNonlocal]->sci));
        free_ocl_buffer(&(nb->plist[eintNonlocal]->cj4));
        free_ocl_buffer(&(nb->plist[eintNonlocal]->excl));
        sfree(nb->plist[eintNonlocal]);
    }

    /* Free nbst */
    ocl_pfree(nb->nbst.e_lj);
    nb->nbst.e_lj = NULL;

    ocl_pfree(nb->nbst.e_el);
    nb->nbst.e_el = NULL;

    ocl_pfree(nb->nbst.fshift);
    nb->nbst.fshift = NULL;

    /* Free debug buffer */
    if (NULL != nb->debug_buffer)
    {
        cl_error = clReleaseMemObject(nb->debug_buffer);
        assert(CL_SUCCESS == cl_error);
        nb->debug_buffer = NULL;
    }

    /* Free command queues */
    clReleaseCommandQueue(nb->stream[eintLocal]);
    nb->stream[eintLocal] = NULL;
    if (nb->bUseTwoStreams)
    {
        clReleaseCommandQueue(nb->stream[eintNonlocal]);
        nb->stream[eintNonlocal] = NULL;
    }
    /* Free other events */
    if (nb->nonlocal_done)
    {
        clReleaseEvent(nb->nonlocal_done);
        nb->nonlocal_done = NULL;
    }
    if (nb->misc_ops_done)
    {
        clReleaseEvent(nb->misc_ops_done);
        nb->misc_ops_done = NULL;
    }

    /* Free timers and timings */
    sfree(nb->timers);
    sfree(nb->timings);
    sfree(nb);

    if (debug)
    {
        fprintf(debug, "Cleaned up OpenCL data structures.\n");
    }
}

/*! OpenCL equivalent of nbnxn_cuda_get_timings */
gmx_wallclock_gpu_t * nbnxn_gpu_get_timings(gmx_nbnxn_ocl_t *nb)
{
    return (nb != NULL && nb->bDoTime) ? nb->timings : NULL;
}

/*! OpenCL equivalent of nbnxn_cuda_reset_timings */
void nbnxn_gpu_reset_timings(nonbonded_verlet_t* nbv)
{
    if (nbv->gpu_nbv && nbv->gpu_nbv->bDoTime)
    {
        init_timings(nbv->gpu_nbv->timings);
    }
}

/*! OpenCL equivalent of nbnxn_cuda_min_ci_balanced */
int nbnxn_gpu_min_ci_balanced(gmx_nbnxn_ocl_t *nb)
{
    return nb != NULL ?
           gpu_min_ci_balanced_factor * nb->dev_info->compute_units : 0;
}

/*! OpenCL equivalent of nbnxn_cuda_is_kernel_ewald_analytical */
gmx_bool nbnxn_gpu_is_kernel_ewald_analytical(const gmx_nbnxn_ocl_t *nb)
{
    return ((nb->nbparam->eeltype == eelOclEWALD_ANA) ||
            (nb->nbparam->eeltype == eelOclEWALD_ANA_TWIN));
}
