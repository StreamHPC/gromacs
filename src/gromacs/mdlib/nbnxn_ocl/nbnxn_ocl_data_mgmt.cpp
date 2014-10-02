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
#include "config.h"

#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "tables.h"
#include "typedefs.h"
#include "types/enums.h"
#include "gromacs/mdlib/nb_verlet.h"
#include "types/interaction_const.h"
#include "types/force_flags.h"
#include "../nbnxn_consts.h"
#include "gmx_detect_hardware.h"

//#include "nbnxn_cuda_types.h"
#include "../nbnxn_ocl/nbnxn_ocl_types.h"
#include "gromacs/mdlib/nbnxn_ocl/nbnxn_ocl_data_mgmt.h"
#include "gpu_utils.h"

#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/common.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/smalloc.h"


static bool bUseCudaEventBlockingSync = false; /* makes the CPU thread block */

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


/* Fw. decl. */
static void nbnxn_cuda_clear_e_fshift(nbnxn_cuda_ptr_t cu_nb);

static int ocl_copy_H2D_generic(cl_mem d_dest, void* h_src, size_t offset, size_t bytes,
                               bool bAsync/* = false*/, cl_command_queue command_queue,
                               cl_event *copy_event)
{
    //cudaError_t stat;
    cl_int cl_error;

    if (d_dest == NULL || h_src == NULL || bytes == 0)
    {
        return -1;
    }

    if (bAsync)
    {
        //stat = cudaMemcpyAsync(d_dest, h_src, bytes, cudaMemcpyHostToDevice, s);
        //CU_RET_ERR(stat, "HtoD cudaMemcpyAsync failed");
        cl_error = clEnqueueWriteBuffer(command_queue, d_dest, CL_FALSE, offset, bytes, h_src, 0, NULL, copy_event);
        assert(cl_error == CL_SUCCESS);
        // TO DO: handle errors
    }
    else
    {
        //stat = cudaMemcpy(d_dest, h_src, bytes, cudaMemcpyHostToDevice);
        //CU_RET_ERR(stat, "HtoD cudaMemcpy failed");
        cl_error = clEnqueueWriteBuffer(command_queue, d_dest, CL_TRUE, offset, bytes, h_src, 0, NULL, copy_event);
        assert(cl_error == CL_SUCCESS);        
        // TO DO: handle errors
    }

    return 0;
}

int ocl_copy_H2D_async(cl_mem d_dest, void * h_src, size_t offset, size_t bytes, cl_command_queue command_queue, cl_event *copy_event)
{
    return ocl_copy_H2D_generic(d_dest, h_src, offset, bytes, true, command_queue, copy_event);
}

int ocl_copy_H2D(cl_mem d_dest, void * h_src, size_t offset, size_t bytes, cl_command_queue command_queue)
{
    return ocl_copy_H2D_generic(d_dest, h_src, offset, bytes, false, command_queue, NULL);
}


int ocl_copy_D2H_generic(void * h_dest, cl_mem d_src, size_t offset, size_t bytes,
                         bool bAsync, cl_command_queue command_queue, cl_event *copy_event)
{
    cl_int cl_error;

    if (h_dest == NULL || d_src == NULL || bytes == 0)
    {
        return -1;
    }

    if (bAsync)
    {        
        cl_error = clEnqueueReadBuffer(command_queue, d_src, CL_FALSE, offset, bytes, h_dest, 0, NULL, copy_event);
        assert(cl_error == CL_SUCCESS);        
        // TO DO: handle errors
    }
    else
    {        
        cl_error = clEnqueueReadBuffer(command_queue, d_src, CL_TRUE, offset, bytes, h_dest, 0, NULL, copy_event);
        assert(cl_error == CL_SUCCESS);        
        // TO DO: handle errors
    }

    return 0;
}

int ocl_copy_D2H_async(void * h_dest, cl_mem d_src, size_t offset, size_t bytes, cl_command_queue command_queue, cl_event *copy_event)
{
    return ocl_copy_D2H_generic(h_dest, d_src, offset, bytes, true, command_queue, copy_event); 
}

/*!
 * If the pointers to the size variables are NULL no resetting happens.
 */
void ocl_free_buffered(cl_mem d_ptr, int *n, int *nalloc)
{
    cl_int cl_error;

    if (d_ptr)
    {
        cl_error = clReleaseMemObject(d_ptr);
        assert(cl_error == CL_SUCCESS);        
        // TO DO: handle errors,
        //stat = cudaFree(d_ptr);
        //CU_RET_ERR(stat, "cudaFree failed");
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
 *  The H2D copy is launched in stream s and can be done synchronously or
 *  asynchronously (the default is the latter).
 */
void ocl_realloc_buffered(cl_mem *d_dest, void *h_src,
                         size_t type_size,
                         int *curr_size, int *curr_alloc_size,
                         int req_size,
                         cl_context context,
                         cl_command_queue s,                         
                         bool bAsync = true)
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

        //stat = cudaMalloc(d_dest, *curr_alloc_size * type_size);
        //CU_RET_ERR(stat, "cudaMalloc failed in cu_free_buffered");

        *d_dest = clCreateBuffer(context, CL_MEM_READ_WRITE, *curr_alloc_size * type_size, NULL, &cl_error);
        assert(cl_error == CL_SUCCESS);
        // TO DO: handle errors, check clCreateBuffer flags
    }

    /* size could have changed without actual reallocation */
    *curr_size = req_size;

    /* upload to device */
    if (h_src)
    {
        if (bAsync)
        {
            ocl_copy_H2D_async(*d_dest, h_src, 0, *curr_size * type_size, s, NULL);
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
 */
static void init_ewald_coulomb_force_table(//cu_nbparam_t          *nbp,
                                           //const cuda_dev_info_t *dev_info
                                           cl_nbparam_t             *nbp,                                           
                                           const ocl_gpu_info_t *dev_info)
{
    float       *ftmp;//, *coul_tab;
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
        //stat = cudaMalloc((void **)&coul_tab, tabsize*sizeof(*coul_tab));
        //CU_RET_ERR(stat, "cudaMalloc failed on coul_tab");
        //coul_tab = clCreateBuffer(dev_info->context, CL_MEM_READ_WRITE, tabsize * sizeof(float), NULL, &cl_error);
        // TO DO: handle errors, check clCreateBuffer flags
        
        cl_image_format array_format;

        array_format.image_channel_data_type = CL_FLOAT;
        array_format.image_channel_order = CL_R;

        coul_tab = clCreateImage2D(dev_info->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            &array_format, tabsize, 1, 0, ftmp, &cl_error);
        assert(cl_error == CL_SUCCESS);
        // TO DO: handle errors

        nbp->coulomb_tab_climg2d = coul_tab;
        nbp->coulomb_tab_size     = tabsize;
        nbp->coulomb_tab_scale    = tabscale; 

////#ifdef TEXOBJ_SUPPORTED
////        /* Only device CC >= 3.0 (Kepler and later) support texture objects */
////        //if (dev_info->prop.major >= 3)
////        if (HAS_CC_3_0_OR_LATER)
////        {
////            cudaResourceDesc rd;
////            memset(&rd, 0, sizeof(rd));
////            rd.resType                  = cudaResourceTypeLinear;
////            rd.res.linear.devPtr        = nbp->coulomb_tab;
////            rd.res.linear.desc.f        = cudaChannelFormatKindFloat;
////            rd.res.linear.desc.x        = 32;
////            rd.res.linear.sizeInBytes   = tabsize*sizeof(*coul_tab);
////
////            cudaTextureDesc td;
////            memset(&td, 0, sizeof(td));
////            td.readMode                 = cudaReadModeElementType;
////            stat = cudaCreateTextureObject(&nbp->coulomb_tab_texobj, &rd, &td, NULL);
////            CU_RET_ERR(stat, "cudaCreateTextureObject on coulomb_tab_texobj failed");
////        }
////        else
////#endif
////        {
////            GMX_UNUSED_VALUE(dev_info);
////            cudaChannelFormatDesc cd   = cudaCreateChannelDesc<float>();
////            stat = cudaBindTexture(NULL, &nbnxn_cuda_get_coulomb_tab_texref(),
////                                   coul_tab, &cd, tabsize*sizeof(*coul_tab));
////            CU_RET_ERR(stat, "cudaBindTexture on coulomb_tab_texref failed");
////        }
    }
////
////    cu_copy_H2D(coul_tab, ftmp, tabsize*sizeof(*coul_tab));
////
////    nbp->coulomb_tab_size     = tabsize;
////    nbp->coulomb_tab_scale    = tabscale;
////
////    pfree(ftmp);
}


/*! Initializes the atomdata structure first time, it only gets filled at
    pair-search. */
static void init_atomdata_first(/*cu_atomdata_t*/cl_atomdata_t *ad, int ntypes, ocl_gpu_info_t *dev_info)
{
    cl_int cl_error;

    ad->ntypes  = ntypes;

    //stat        = cudaMalloc((void**)&ad->shift_vec, SHIFTS*sizeof(*ad->shift_vec));
    //CU_RET_ERR(stat, "cudaMalloc failed on ad->shift_vec");
    ad->shift_vec = clCreateBuffer(dev_info->context, CL_MEM_READ_WRITE, SHIFTS * sizeof(rvec), NULL, &cl_error);        
    assert(cl_error == CL_SUCCESS);
    ad->bShiftVecUploaded = false;
    // TO DO: handle errors, check clCreateBuffer flags

    //stat = cudaMalloc((void**)&ad->fshift, SHIFTS*sizeof(*ad->fshift));
    //CU_RET_ERR(stat, "cudaMalloc failed on ad->fshift");
    ad->fshift = clCreateBuffer(dev_info->context, CL_MEM_READ_WRITE, SHIFTS * sizeof(rvec), NULL, &cl_error);
    assert(cl_error == CL_SUCCESS);    
    // TO DO: handle errors, check clCreateBuffer flags

    //stat = cudaMalloc((void**)&ad->e_lj, sizeof(*ad->e_lj));
    //CU_RET_ERR(stat, "cudaMalloc failed on ad->e_lj");
    ad->e_lj = clCreateBuffer(dev_info->context, CL_MEM_READ_WRITE, sizeof(float), NULL, &cl_error);
    assert(cl_error == CL_SUCCESS);    
    // TO DO: handle errors, check clCreateBuffer flags

    //stat = cudaMalloc((void**)&ad->e_el, sizeof(*ad->e_el));
    //CU_RET_ERR(stat, "cudaMalloc failed on ad->e_el");
    ad->e_el = clCreateBuffer(dev_info->context, CL_MEM_READ_WRITE, sizeof(float), NULL, &cl_error);
    assert(cl_error == CL_SUCCESS);    
    // TO DO: handle errors, check clCreateBuffer flags

    /* initialize to NULL poiters to data that is not allocated here and will
       need reallocation in nbnxn_cuda_init_atomdata */
    ad->xq = NULL;
    ad->f  = NULL;

    /* size -1 indicates that the respective array hasn't been initialized yet */
    ad->natoms = -1;
    ad->nalloc = -1;
}

/*! Selects the Ewald kernel type, analytical on SM 3.0 and later, tabulated on
    earlier GPUs, single or twin cut-off. */
static int pick_ewald_kernel_type(bool bTwinCut)
{
    bool bUseAnalyticalEwald, bForceAnalyticalEwald, bForceTabulatedEwald;
    int  kernel_type;

    /* Benchmarking/development environment variables to force the use of
       analytical or tabulated Ewald kernel. */
    bForceAnalyticalEwald = (getenv("GMX_CUDA_NB_ANA_EWALD") != NULL);
    bForceTabulatedEwald  = (getenv("GMX_CUDA_NB_TAB_EWALD") != NULL);

    if (bForceAnalyticalEwald && bForceTabulatedEwald)
    {
        gmx_incons("Both analytical and tabulated Ewald CUDA non-bonded kernels "
                   "requested through environment variables.");
    }

    /* By default, on SM 3.0 and later use analytical Ewald, on earlier tabulated. */
    //if ((dev_info->prop.major >= 3 || bForceAnalyticalEwald) && !bForceTabulatedEwald)
    if ((HAS_CC_3_0_OR_LATER || bForceAnalyticalEwald) && !bForceTabulatedEwald)    
    {
        bUseAnalyticalEwald = true;

        if (debug)
        {
            fprintf(debug, "Using analytical Ewald CUDA kernels\n");
        }
    }
    else
    {
        bUseAnalyticalEwald = false;

        if (debug)
        {
            fprintf(debug, "Using tabulated Ewald CUDA kernels\n");
        }
    }

    /* Use twin cut-off kernels if requested by bTwinCut or the env. var.
       forces it (use it for debugging/benchmarking only). */
    if (!bTwinCut && (getenv("GMX_CUDA_NB_EWALD_TWINCUT") == NULL))
    {
        kernel_type = bUseAnalyticalEwald ? eelOclEWALD_ANA : eelOclEWALD_TAB;
    }
    else
    {
        kernel_type = bUseAnalyticalEwald ? eelOclEWALD_ANA_TWIN : eelOclEWALD_TAB_TWIN;
    }

    return kernel_type;
}

/*! Copies all parameters related to the cut-off from ic to nbp */
static void set_cutoff_parameters(//cu_nbparam_t              *nbp,
                                  cl_nbparam_t              *nbp,
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

/*! Initializes the nonbonded parameter data structure. */
static void init_nbparam(/*cu_nbparam_t*/cl_nbparam_t  *nbp,
                         const interaction_const_t *ic,
                         const nbnxn_atomdata_t    *nbat,
                         const /*cuda_dev_info_t*/ocl_gpu_info_t     *dev_info)
{
    int         ntypes, nnbfp, nnbfp_comb;
    cl_int      cl_error;


    ntypes  = nbat->ntype;

    set_cutoff_parameters(nbp, ic);

    if (ic->vdwtype == evdwCUT)
    {
        switch (ic->vdw_modifier)
        {
            case eintmodNONE:
            case eintmodPOTSHIFT:
                nbp->vdwtype = evdwOclCUT;
                break;
            case eintmodFORCESWITCH:
                nbp->vdwtype = evdwOclFSWITCH;
                break;
            case eintmodPOTSWITCH:
                nbp->vdwtype = evdwOclPSWITCH;
                break;
            default:
                gmx_incons("The requested VdW interaction modifier is not implemented in the CUDA GPU accelerated kernels!");
                break;
        }
    }
    else if (ic->vdwtype == evdwPME)
    {
        if (ic->ljpme_comb_rule == ljcrGEOM)
        {
            assert(nbat->comb_rule == ljcrGEOM);
            nbp->vdwtype = evdwOclEWALDGEOM;
        }
        else
        {
            assert(nbat->comb_rule == ljcrLB);
            nbp->vdwtype = evdwOclEWALDLB;
        }
    }
    else
    {
        gmx_incons("The requested VdW type is not implemented in the CUDA GPU accelerated kernels!");
    }

    if (ic->eeltype == eelCUT)
    {
        nbp->eeltype = eelOclCUT;
    }
    else if (EEL_RF(ic->eeltype))
    {
        nbp->eeltype = eelOclRF;
    }
    else if ((EEL_PME(ic->eeltype) || ic->eeltype == eelEWALD))
    {
        /* Initially rcoulomb == rvdw, so it's surely not twin cut-off. */
        nbp->eeltype = pick_ewald_kernel_type(false);
    }
    else
    {
        /* Shouldn't happen, as this is checked when choosing Verlet-scheme */
        gmx_incons("The requested electrostatics type is not implemented in the CUDA GPU accelerated kernels!");
    }

    /* generate table for PME */
    nbp->coulomb_tab_climg2d = NULL;
    if (nbp->eeltype == eelOclEWALD_TAB || nbp->eeltype == eelOclEWALD_TAB_TWIN)
    {
        init_ewald_coulomb_force_table(nbp, dev_info);
    }
    else
    // TO DO: improvement needed.
    // The image2d is created here even if eeltype is not eelCuEWALD_TAB or eelCuEWALD_TAB_TWIN because the OpenCL kernels
    // don't accept NULL values for image2D parameters.
    {
        cl_image_format array_format;

        array_format.image_channel_data_type = CL_FLOAT;
        array_format.image_channel_order = CL_R;

        nbp->coulomb_tab_climg2d = clCreateImage2D(dev_info->context, CL_MEM_READ_WRITE,
            &array_format, 1, 1, 0, NULL, &cl_error);
        // TO DO: handle errors        
    }

    nnbfp      = 2*ntypes*ntypes;
    nnbfp_comb = 2*ntypes;

    ////////////////////////////////////////////////////////////
    // In the CUDA implementation, the code below was creating two buffers and then binding two textures
    // to the buffers.
    // With OpenCL we can just create two Image2D objects.
    ////////////////////////////////////////////////////////////


    //stat  = cudaMalloc((void **)&nbp->nbfp, nnbfp*sizeof(*nbp->nbfp));
    //CU_RET_ERR(stat, "cudaMalloc failed on nbp->nbfp");
    //cu_copy_H2D(nbp->nbfp, nbat->nbfp, nnbfp*sizeof(*nbp->nbfp));

    ////nbp->nbfp = clCreateBuffer(dev_info->context, CL_MEM_READ_WRITE, nnbfp * sizeof(float), NULL, &cl_error);
    ////// TO DO: handle errors, check clCreateBuffer flags
    ////cl_error = clEnqueueWriteBuffer(dev_info->command_queue, nbp->nbfp, CL_TRUE,
				////	0, (size_t)(nnbfp * sizeof(float)), (void*)(nbat->nbfp), 0, NULL, NULL);
    ////// TO DO: handle errors


    if (ic->vdwtype == evdwPME)
    {
        //stat  = cudaMalloc((void **)&nbp->nbfp_comb, nnbfp_comb*sizeof(*nbp->nbfp_comb));
        //CU_RET_ERR(stat, "cudaMalloc failed on nbp->nbfp_comb");
        //cu_copy_H2D(nbp->nbfp_comb, nbat->nbfp_comb, nnbfp_comb*sizeof(*nbp->nbfp_comb));

        
        ////    nbp->nbfp_comb = clCreateBuffer(dev_info->context, CL_MEM_READ_WRITE, nnbfp_comb * sizeof(float), NULL, &cl_error);
        ////    // TO DO: handle errors, check clCreateBuffer flags
        ////    cl_error = clEnqueueWriteBuffer(dev_info->command_queue, nbp->nbfp_comb, CL_TRUE,
				    ////0, (size_t)(nnbfp_comb * sizeof(float)), (void*)(nbat->nbfp_comb), 0, NULL, NULL);
        ////    // TO DO: handle errors
    }

    {
        cl_image_format array_format;

        array_format.image_channel_data_type = CL_FLOAT;
        array_format.image_channel_order = CL_R;

        nbp->nbfp_climg2d = clCreateImage2D(dev_info->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            &array_format, nnbfp, 1, 0, nbat->nbfp, &cl_error);
        assert(cl_error == CL_SUCCESS);
        // TO DO: handle errors

        if (ic->vdwtype == evdwPME)
        {
            nbp->nbfp_comb_climg2d = clCreateImage2D(dev_info->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                &array_format, nnbfp_comb, 1, 0, nbat->nbfp_comb, &cl_error);
            assert(cl_error == CL_SUCCESS);            
            // TO DO: handle errors
        }
        else
        {
            // TO DO: improvement needed.
            // The image2d is created here even if vdwtype is not evdwPME because the OpenCL kernels
            // don't accept NULL values for image2D parameters.
            nbp->nbfp_comb_climg2d = clCreateImage2D(dev_info->context, CL_MEM_READ_WRITE,
                &array_format, 1, 1, 0, NULL, &cl_error);
            assert(cl_error == CL_SUCCESS);
            // TO DO: handle errors
        }
    }

////#ifdef TEXOBJ_SUPPORTED
////    /* Only device CC >= 3.0 (Kepler and later) support texture objects */
////    if (dev_info->prop.major >= 3)
////    {
////        cudaResourceDesc rd;
////        cudaTextureDesc  td;
////
////        memset(&rd, 0, sizeof(rd));
////        rd.resType                  = cudaResourceTypeLinear;
////        rd.res.linear.devPtr        = nbp->nbfp;
////        rd.res.linear.desc.f        = cudaChannelFormatKindFloat;
////        rd.res.linear.desc.x        = 32;
////        rd.res.linear.sizeInBytes   = nnbfp*sizeof(*nbp->nbfp);
////
////        memset(&td, 0, sizeof(td));
////        td.readMode                 = cudaReadModeElementType;
////        stat = cudaCreateTextureObject(&nbp->nbfp_texobj, &rd, &td, NULL);
////        CU_RET_ERR(stat, "cudaCreateTextureObject on nbfp_texobj failed");
////
////        if (ic->vdwtype == evdwPME)
////        {
////            memset(&rd, 0, sizeof(rd));
////            rd.resType                  = cudaResourceTypeLinear;
////            rd.res.linear.devPtr        = nbp->nbfp_comb;
////            rd.res.linear.desc.f        = cudaChannelFormatKindFloat;
////            rd.res.linear.desc.x        = 32;
////            rd.res.linear.sizeInBytes   = nnbfp_comb*sizeof(*nbp->nbfp_comb);
////
////            memset(&td, 0, sizeof(td));
////            td.readMode = cudaReadModeElementType;
////            stat        = cudaCreateTextureObject(&nbp->nbfp_comb_texobj, &rd, &td, NULL);
////            CU_RET_ERR(stat, "cudaCreateTextureObject on nbfp_comb_texobj failed");
////        }
////    }
////    else
////#endif
////    {
////        cudaChannelFormatDesc cd = cudaCreateChannelDesc<float>();
////        stat = cudaBindTexture(NULL, &nbnxn_cuda_get_nbfp_texref(),
////                               nbp->nbfp, &cd, nnbfp*sizeof(*nbp->nbfp));
////        CU_RET_ERR(stat, "cudaBindTexture on nbfp_texref failed");
////
////        if (ic->vdwtype == evdwPME)
////        {
////            stat = cudaBindTexture(NULL, &nbnxn_cuda_get_nbfp_comb_texref(),
////                                   nbp->nbfp_comb, &cd, nnbfp_comb*sizeof(*nbp->nbfp_comb));
////            CU_RET_ERR(stat, "cudaBindTexture on nbfp_comb_texref failed");
////        }
////    }
}

///*! Re-generate the GPU Ewald force table, resets rlist, and update the
// *  electrostatic type switching to twin cut-off (or back) if needed. */
//void nbnxn_cuda_pme_loadbal_update_param(const nonbonded_verlet_t    *nbv,
//                                         const interaction_const_t   *ic)
//{
//    if (!nbv || nbv->grp[0].kernel_type != nbnxnk8x8x8_CUDA)
//    {
//        return;
//    }
//    nbnxn_cuda_ptr_t cu_nb = nbv->cu_nbv;
//    cu_nbparam_t    *nbp   = cu_nb->nbparam;
//
//    set_cutoff_parameters(nbp, ic);
//
//    nbp->eeltype        = pick_ewald_kernel_type(ic->rcoulomb != ic->rvdw,
//                                                 cu_nb->dev_info);
//
//    init_ewald_coulomb_force_table(cu_nb->nbparam, cu_nb->dev_info);
//}

/*! Initializes the pair list data structure. */
static void init_plist(cl_plist_t *pl)
{
    /* initialize to NULL pointers to data that is not allocated here and will
       need reallocation in nbnxn_cuda_init_pairlist */
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

/*! Initializes the timer data structure. */
static void init_timers(cl_timers_t *t, bool bUseTwoStreams)
{
    ////cudaError_t stat;
    ////int         eventflags = ( bUseCudaEventBlockingSync ? cudaEventBlockingSync : cudaEventDefault );

    ////stat = cudaEventCreateWithFlags(&(t->start_atdat), eventflags);
    ////CU_RET_ERR(stat, "cudaEventCreate on start_atdat failed");
    ////stat = cudaEventCreateWithFlags(&(t->stop_atdat), eventflags);
    ////CU_RET_ERR(stat, "cudaEventCreate on stop_atdat failed");

    // TO DO: update for OpenCL

    /////* The non-local counters/stream (second in the array) are needed only with DD. */
    ////for (int i = 0; i <= (bUseTwoStreams ? 1 : 0); i++)
    ////{
    ////    stat = cudaEventCreateWithFlags(&(t->start_nb_k[i]), eventflags);
    ////    CU_RET_ERR(stat, "cudaEventCreate on start_nb_k failed");
    ////    stat = cudaEventCreateWithFlags(&(t->stop_nb_k[i]), eventflags);
    ////    CU_RET_ERR(stat, "cudaEventCreate on stop_nb_k failed");


    ////    stat = cudaEventCreateWithFlags(&(t->start_pl_h2d[i]), eventflags);
    ////    CU_RET_ERR(stat, "cudaEventCreate on start_pl_h2d failed");
    ////    stat = cudaEventCreateWithFlags(&(t->stop_pl_h2d[i]), eventflags);
    ////    CU_RET_ERR(stat, "cudaEventCreate on stop_pl_h2d failed");

    ////    stat = cudaEventCreateWithFlags(&(t->start_nb_h2d[i]), eventflags);
    ////    CU_RET_ERR(stat, "cudaEventCreate on start_nb_h2d failed");
    ////    stat = cudaEventCreateWithFlags(&(t->stop_nb_h2d[i]), eventflags);
    ////    CU_RET_ERR(stat, "cudaEventCreate on stop_nb_h2d failed");

    ////    stat = cudaEventCreateWithFlags(&(t->start_nb_d2h[i]), eventflags);
    ////    CU_RET_ERR(stat, "cudaEventCreate on start_nb_d2h failed");
    ////    stat = cudaEventCreateWithFlags(&(t->stop_nb_d2h[i]), eventflags);
    ////    CU_RET_ERR(stat, "cudaEventCreate on stop_nb_d2h failed");
    ////}
}

/*! Initializes the timings data structure. */
static void init_timings(wallclock_gpu_t *t)
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

void nbnxn_init_kernels(nbnxn_opencl_ptr_t  nb)
{
    cl_int cl_error;

    /* Init to 0 main kernel arrays */
    /* They will be later on initialized in select_nbnxn_kernel */
    memset(nb->kernel_ener_noprune_ptr, 0, sizeof(nb->kernel_ener_noprune_ptr));
    memset(nb->kernel_ener_prune_ptr, 0, sizeof(nb->kernel_ener_prune_ptr));
    memset(nb->kernel_noener_noprune_ptr, 0, sizeof(nb->kernel_noener_noprune_ptr));
    memset(nb->kernel_noener_prune_ptr, 0, sizeof(nb->kernel_noener_prune_ptr));

    /* Init auxiliary kernels */
    nb->kernel_memset_f = clCreateKernel(nb->dev_info->program,"memset_f", &cl_error);
    assert(cl_error == CL_SUCCESS); 

    nb->kernel_memset_f2 = clCreateKernel(nb->dev_info->program,"memset_f2", &cl_error);
    assert(cl_error == CL_SUCCESS); 

    nb->kernel_memset_f3 = clCreateKernel(nb->dev_info->program,"memset_f3", &cl_error);
    assert(cl_error == CL_SUCCESS); 

    nb->kernel_zero_e_fshift = clCreateKernel(nb->dev_info->program,"zero_e_fshift", &cl_error);
    assert(cl_error == CL_SUCCESS);    
}

void nbnxn_ocl_init(FILE                 *fplog,
                     nbnxn_opencl_ptr_t     *p_cu_nb,
                     const gmx_gpu_info_t *gpu_info,
                     const gmx_gpu_opt_t  *gpu_opt,
                     int                   my_gpu_index,
                     gmx_bool              bLocalAndNonlocal)
{
    //nbnxn_cuda_ptr_t  nb;
    nbnxn_opencl_ptr_t  nb;
    cl_int            cl_error;    
    char              sbuf[STRLEN];
    bool              bStreamSync, bNoStreamSync, bTMPIAtomics, bX86, bOldDriver;
    cl_command_queue_properties queue_properties;

    assert(gpu_info);

    if (p_cu_nb == NULL)
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
    nb->dev_info = &gpu_info->ocl_dev[my_gpu_index];
        //&gpu_info->cuda_dev[get_gpu_device_id(gpu_info, gpu_opt, my_gpu_index)];

    /* init the kernels */
    nbnxn_init_kernels(nb);

    /* init nbst */
    ocl_pmalloc((void**)&nb->nbst.e_lj, sizeof(*nb->nbst.e_lj));
    ocl_pmalloc((void**)&nb->nbst.e_el, sizeof(*nb->nbst.e_el));
    ocl_pmalloc((void**)&nb->nbst.fshift, SHIFTS * sizeof(*nb->nbst.fshift));

    init_plist(nb->plist[eintLocal]);

    if (nb->bDoTime)
        queue_properties = CL_QUEUE_PROFILING_ENABLE;
    else
        queue_properties = 0;

    /* local/non-local GPU streams */
    nb->stream[eintLocal] = clCreateCommandQueue(nb->dev_info->context, nb->dev_info->ocl_gpu_id.ocl_device_id, queue_properties, &cl_error);
    assert(cl_error == CL_SUCCESS);
    // TO DO: check for errors        
    //stat = cudaStreamCreate(&nb->stream[eintLocal]);
    //CU_RET_ERR(stat, "cudaStreamCreate on stream[eintLocal] failed");
    if (nb->bUseTwoStreams)
    {
        init_plist(nb->plist[eintNonlocal]);

        /* CUDA stream priority available in the CUDA RT 5.5 API.
         * Note that the device we're running on does not have to support
         * priorities, because we are querying the priority range which in this
         * case will be a single value.
         */
#if CUDA_VERSION >= 5500
        {
            int highest_priority;
            stat = cudaDeviceGetStreamPriorityRange(NULL, &highest_priority);
            CU_RET_ERR(stat, "cudaDeviceGetStreamPriorityRange failed");

            stat = cudaStreamCreateWithPriority(&nb->stream[eintNonlocal],
                                                cudaStreamDefault,
                                                highest_priority);
            CU_RET_ERR(stat, "cudaStreamCreateWithPriority on stream[eintNonlocal] failed");
        }
#else
        //stat = cudaStreamCreate(&nb->stream[eintNonlocal]);
        nb->stream[eintNonlocal] = clCreateCommandQueue(nb->dev_info->context, nb->dev_info->ocl_gpu_id.ocl_device_id, queue_properties, &cl_error);
        assert(cl_error == CL_SUCCESS);        
        // TO DO: check for errors        
        //CU_RET_ERR(stat, "cudaStreamCreate on stream[eintNonlocal] failed");
#endif
    }

    /* init events for sychronization (timing disabled for performance reasons!) */
    nb->nonlocal_done = clCreateUserEvent(nb->dev_info->context, &cl_error);
    assert(cl_error == CL_SUCCESS);    
    // TO DO: check for errors
    //stat = cudaEventCreateWithFlags(&nb->nonlocal_done, cudaEventDisableTiming);
    //CU_RET_ERR(stat, "cudaEventCreate on nonlocal_done failed");

    nb->misc_ops_done = clCreateUserEvent(nb->dev_info->context, &cl_error);
    assert(cl_error == CL_SUCCESS);    
    // TO DO: check for errors
    //stat = cudaEventCreateWithFlags(&nb->misc_ops_done, cudaEventDisableTiming);
    //CU_RET_ERR(stat, "cudaEventCreate on misc_ops_one failed");

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

    bStreamSync    = getenv("GMX_CUDA_STREAMSYNC") != NULL;
    bNoStreamSync  = getenv("GMX_NO_CUDA_STREAMSYNC") != NULL;

#ifdef TMPI_ATOMICS
    bTMPIAtomics = true;
#else
    bTMPIAtomics = false;
#endif

#ifdef GMX_TARGET_X86
    bX86 = true;
#else
    bX86 = false;
#endif

    if (bStreamSync && bNoStreamSync)
    {
        gmx_fatal(FARGS, "Conflicting environment variables: both GMX_CUDA_STREAMSYNC and GMX_NO_CUDA_STREAMSYNC defined");
    }

/*
    stat = cudaDriverGetVersion(&cuda_drv_ver);
    CU_RET_ERR(stat, "cudaDriverGetVersion failed");

    bOldDriver = (cuda_drv_ver < 5000);
*/
    bOldDriver = CL_FALSE; //??

    // TO DO: fix the below code for OpenCL
    //////////if ((nb->dev_info->prop.ECCEnabled == 1) && bOldDriver)
    //////////{
    //////////    /* Polling wait should be used instead of cudaStreamSynchronize only if:
    //////////     *   - ECC is ON & driver is old (checked above),
    //////////     *   - we're on x86/x86_64,
    //////////     *   - atomics are available, and
    //////////     *   - GPUs are not being shared.
    //////////     */
    //////////    bool bShouldUsePollSync = (bX86 && bTMPIAtomics &&
    //////////                               (gmx_count_gpu_dev_shared(gpu_opt) < 1));

    //////////    if (bStreamSync)
    //////////    {
    //////////        nb->bUseStreamSync = true;

    //////////        /* only warn if polling should be used */
    //////////        if (bShouldUsePollSync)
    //////////        {
    //////////            md_print_warn(fplog,
    //////////                          "NOTE: Using a GPU with ECC enabled and CUDA driver API version <5.0, but\n"
    //////////                          "      cudaStreamSynchronize waiting is forced by the GMX_CUDA_STREAMSYNC env. var.\n");
    //////////        }
    //////////    }
    //////////    else
    //////////    {
    //////////        nb->bUseStreamSync = !bShouldUsePollSync;

    //////////        if (bShouldUsePollSync)
    //////////        {
    //////////            md_print_warn(fplog,
    //////////                          "NOTE: Using a GPU with ECC enabled and CUDA driver API version <5.0, known to\n"
    //////////                          "      cause performance loss. Switching to the alternative polling GPU wait.\n"
    //////////                          "      If you encounter issues, switch back to standard GPU waiting by setting\n"
    //////////                          "      the GMX_CUDA_STREAMSYNC environment variable.\n");
    //////////        }
    //////////        else
    //////////        {
    //////////            /* Tell the user that the ECC+old driver combination can be bad */
    //////////            sprintf(sbuf,
    //////////                    "NOTE: Using a GPU with ECC enabled and CUDA driver API version <5.0.\n"
    //////////                    "      A known bug in this driver version can cause performance loss.\n"
    //////////                    "      However, the polling wait workaround can not be used because\n%s\n"
    //////////                    "      Consider updating the driver or turning ECC off.",
    //////////                    (bX86 && bTMPIAtomics) ?
    //////////                    "      GPU(s) are being oversubscribed." :
    //////////                    "      atomic operations are not supported by the platform/CPU+compiler.");
    //////////            md_print_warn(fplog, sbuf);
    //////////        }
    //////////    }
    //////////}
    //////////else
    //////////{
    //////////    if (bNoStreamSync)
    //////////    {
    //////////        nb->bUseStreamSync = false;

    //////////        md_print_warn(fplog,
    //////////                      "NOTE: Polling wait for GPU synchronization requested by GMX_NO_CUDA_STREAMSYNC\n");
    //////////    }
    //////////    else
    //////////    {
    //////////        /* no/off ECC, cudaStreamSynchronize not turned off by env. var. */
    //////////        nb->bUseStreamSync = true;
    //////////    }
    //////////}

    ///////////* CUDA timing disabled as event timers don't work:
    //////////   - with multiple streams = domain-decomposition;
    //////////   - with the polling waiting hack (without cudaStreamSynchronize);
    //////////   - when turned off by GMX_DISABLE_CUDA_TIMING.
    ////////// */
    //////////nb->bDoTime = (!nb->bUseTwoStreams && nb->bUseStreamSync &&
    //////////               (getenv("GMX_DISABLE_CUDA_TIMING") == NULL));

    if (nb->bDoTime)
    {
        init_timers(nb->timers, nb->bUseTwoStreams);
        init_timings(nb->timings);
    }

    ///////////* set the kernel type for the current GPU */
    ///////////* pick L1 cache configuration */
    //////////nbnxn_cuda_set_cacheconfig(nb->dev_info);

    *p_cu_nb = nb;

    if (debug)
    {
        fprintf(debug, "Initialized CUDA data structures.\n");
    }
}

/*! Clears nonbonded shift force output array and energy outputs on the GPU. */
static void 
nbnxn_ocl_clear_e_fshift(nbnxn_opencl_ptr_t ocl_nb)
{
    
    cl_int               cl_error = CL_SUCCESS;
    cl_atomdata_t *      adat     = ocl_nb->atdat;
    cl_command_queue     ls       = ocl_nb->stream[eintLocal];    
    ocl_gpu_info_t *     dev_info = ocl_nb->dev_info;
    
    size_t               dim_block[3] = {1,1,1} ;
    size_t               dim_grid[3]  = {1,1,1};
    cl_int               shifts       = SHIFTS*3;
    
    cl_int               arg_no;    
    
    cl_kernel            zero_e_fshift = ocl_nb->kernel_zero_e_fshift;    
    
    dim_block[0] = 64;
    dim_grid[0]  = ((shifts/64)*64) + ((shifts%64)?64:0) ;

    arg_no = 0;    
    cl_error = clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_mem), &(adat->fshift));    
    cl_error |= clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_mem), &(adat->e_lj));     
    cl_error |= clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_mem), &(adat->e_el));     
    cl_error |= clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_uint), &shifts);         
    assert(cl_error == CL_SUCCESS);
    
    cl_error = clEnqueueNDRangeKernel(ls, zero_e_fshift, 3, NULL, dim_grid, dim_block, 0, NULL, NULL);    
    //cl_error |= clFinish(ls);        
    assert(cl_error == CL_SUCCESS);    
    
}

/*! Clears the first natoms_clear elements of the GPU nonbonded force output array. */
static void nbnxn_ocl_clear_f(nbnxn_opencl_ptr_t ocl_nb, int natoms_clear)
{
    
    cl_int               cl_error = CL_SUCCESS;
    cl_atomdata_t *      adat     = ocl_nb->atdat;
    cl_command_queue     ls       = ocl_nb->stream[eintLocal];    
    ocl_gpu_info_t *     dev_info = ocl_nb->dev_info;
    cl_float             value    = 0.0f;
    
    size_t               dim_block[3] = {1,1,1} ;
    size_t               dim_grid[3]  = {1,1,1};
    
    cl_int               arg_no;
    
    cl_kernel            memset_f = ocl_nb->kernel_memset_f3;       
    
    dim_block[0] = 64;
    dim_grid[0]  = ((natoms_clear/64)*64) + ((natoms_clear%64)?64:0) ;
    
    arg_no = 0;    
    cl_error = clSetKernelArg(memset_f, arg_no++, sizeof(cl_mem), &(adat->f));      
    cl_error = clSetKernelArg(memset_f, arg_no++, sizeof(cl_float), &value);      
    cl_error |= clSetKernelArg(memset_f, arg_no++, sizeof(cl_uint), &natoms_clear);         
    assert(cl_error == CL_SUCCESS);
    
    cl_error = clEnqueueNDRangeKernel(ls, memset_f, 3, NULL, dim_grid, dim_block, 0, NULL, NULL);    
    //cl_error |= clFinish(ls);        
    assert(cl_error == CL_SUCCESS);

    //stat = cudaMemsetAsync(adat->f, 0, natoms_clear * sizeof(*adat->f), ls);
    //CU_RET_ERR(stat, "cudaMemsetAsync on f falied");
}

void 
nbnxn_ocl_clear_outputs(nbnxn_opencl_ptr_t ocl_nb, 
                        int flags)
{
    nbnxn_ocl_clear_f(ocl_nb, ocl_nb->atdat->natoms);
    /* clear shift force array and energies if the outputs were
       used in the current step */
    if (flags & GMX_FORCE_VIRIAL)
    {
        nbnxn_ocl_clear_e_fshift(ocl_nb);
    }
}

void nbnxn_ocl_init_const(nbnxn_opencl_ptr_t                ocl_nb,
                           const interaction_const_t      *ic,
                           const nonbonded_verlet_group_t *nbv_group)
{
    // TO DO: add proper code
    ////init_atomdata_first(cu_nb->atdat, nbv_group[0].nbat->ntype);
    init_atomdata_first(ocl_nb->atdat, nbv_group[0].nbat->ntype, ocl_nb->dev_info);

    ////init_nbparam(cu_nb->nbparam, ic, nbv_group[0].nbat, cu_nb->dev_info);
    init_nbparam(ocl_nb->nbparam, ic, nbv_group[0].nbat, ocl_nb->dev_info);

    /* clear energy and shift force outputs */
    nbnxn_ocl_clear_e_fshift(ocl_nb);
}

void nbnxn_ocl_init_pairlist(nbnxn_opencl_ptr_t        ocl_nb,
                              const nbnxn_pairlist_t *h_plist,
                              int                     iloc)
{
    char          sbuf[STRLEN];
    bool          bDoTime    = ocl_nb->bDoTime;
    //cudaStream_t  stream     = cu_nb->stream[iloc];
    cl_command_queue stream     = ocl_nb->stream[iloc];
    cl_plist_t   *d_plist    = ocl_nb->plist[iloc];

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

    if (bDoTime)
    {
        // TO DO: add missing implementation
        //stat = cudaEventRecord(cu_nb->timers->start_pl_h2d[iloc], stream);
        //CU_RET_ERR(stat, "cudaEventRecord failed");
    }

    //cu_realloc_buffered((void **)&d_plist->sci, h_plist->sci, sizeof(*d_plist->sci),
    //                    &d_plist->nsci, &d_plist->sci_nalloc,
    //                    h_plist->nsci,
    //                    stream, true);
    ocl_realloc_buffered(&d_plist->sci, h_plist->sci, sizeof(nbnxn_sci_t),
                        &d_plist->nsci, &d_plist->sci_nalloc,
                        h_plist->nsci,
                        ocl_nb->dev_info->context,
                        stream, true);

    //cu_realloc_buffered((void **)&d_plist->cj4, h_plist->cj4, sizeof(*d_plist->cj4),
    //                    &d_plist->ncj4, &d_plist->cj4_nalloc,
    //                    h_plist->ncj4,
    //                    stream, true);
    ocl_realloc_buffered(&d_plist->cj4, h_plist->cj4, sizeof(nbnxn_cj4_t),
                        &d_plist->ncj4, &d_plist->cj4_nalloc,
                        h_plist->ncj4,
                        ocl_nb->dev_info->context,
                        stream, true);

    //cu_realloc_buffered((void **)&d_plist->excl, h_plist->excl, sizeof(*d_plist->excl),
    //                    &d_plist->nexcl, &d_plist->excl_nalloc,
    //                    h_plist->nexcl,
    //                    stream, true);
    ocl_realloc_buffered(&d_plist->excl, h_plist->excl, sizeof(nbnxn_excl_t),
                        &d_plist->nexcl, &d_plist->excl_nalloc,
                        h_plist->nexcl,
                        ocl_nb->dev_info->context,
                        stream, true);

    if (bDoTime)
    {
        // TO DO: add missing implementation
        //stat = cudaEventRecord(cu_nb->timers->stop_pl_h2d[iloc], stream);
        //CU_RET_ERR(stat, "cudaEventRecord failed");
    }

    /* need to prune the pair list during the next step */
    d_plist->bDoPrune = true;
}

void nbnxn_ocl_upload_shiftvec(nbnxn_opencl_ptr_t        ocl_nb,
                                const nbnxn_atomdata_t *nbatom)
{
    cl_atomdata_t *adat  = ocl_nb->atdat;
    cl_command_queue ls    = ocl_nb->stream[eintLocal];

    /* only if we have a dynamic box */
    if (nbatom->bDynamicBox || !adat->bShiftVecUploaded)
    {
        ocl_copy_H2D_async(adat->shift_vec, nbatom->shift_vec, 0,
                          SHIFTS * sizeof(rvec), ls, NULL);
        adat->bShiftVecUploaded = true;
    }
}

void nbnxn_ocl_init_atomdata(nbnxn_opencl_ptr_t        ocl_nb,
                              const nbnxn_atomdata_t *nbat)
{
    cl_int         cl_error;
    int            nalloc, natoms;
    bool           realloced;
    bool           bDoTime   = ocl_nb->bDoTime;
    //cu_timers_t   *timers    = ocl_nb->timers;
    cl_timers_t *timers = ocl_nb->timers;

    //cu_atomdata_t *d_atdat   = ocl_nb->atdat;
    cl_atomdata_t *d_atdat   = ocl_nb->atdat;
    //cudaStream_t   ls        = ocl_nb->stream[eintLocal];
    cl_command_queue ls        = ocl_nb->stream[eintLocal];

    natoms    = nbat->natoms;
    realloced = false;

    //if (bDoTime)
    //{
    //    /* time async copy */
    //    stat = cudaEventRecord(timers->start_atdat, ls);
    //    CU_RET_ERR(stat, "cudaEventRecord failed");
    //}

    /* need to reallocate if we have to copy more atoms than the amount of space
       available and only allocate if we haven't initialized yet, i.e d_atdat->natoms == -1 */
    if (natoms > d_atdat->nalloc)
    {
        nalloc = over_alloc_small(natoms);

        /* free up first if the arrays have already been initialized */
        if (d_atdat->nalloc != -1)
        {
            //cu_free_buffered(d_atdat->f, &d_atdat->natoms, &d_atdat->nalloc);
            //cu_free_buffered(d_atdat->xq);
            //cu_free_buffered(d_atdat->atom_types);

            ocl_free_buffered(d_atdat->f, &d_atdat->natoms, &d_atdat->nalloc);
            ocl_free_buffered(d_atdat->xq, NULL, NULL);
            ocl_free_buffered(d_atdat->atom_types, NULL, NULL);
        }

        //stat = cudaMalloc((void **)&d_atdat->f, nalloc*sizeof(*d_atdat->f));
        //CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->f");
        d_atdat->f = clCreateBuffer(ocl_nb->dev_info->context, CL_MEM_READ_WRITE, nalloc * sizeof(rvec), NULL, &cl_error);
        // TO DO: handle errors, check clCreateBuffer flags

        //stat = cudaMalloc((void **)&d_atdat->xq, nalloc*sizeof(*d_atdat->xq));
        //CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->xq");
        d_atdat->xq = clCreateBuffer(ocl_nb->dev_info->context, CL_MEM_READ_WRITE, nalloc * sizeof(cl_float4), NULL, &cl_error);
        // TO DO: handle errors, check clCreateBuffer flags

        //stat = cudaMalloc((void **)&d_atdat->atom_types, nalloc*sizeof(*d_atdat->atom_types));
        //CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->atom_types");
        d_atdat->atom_types = clCreateBuffer(ocl_nb->dev_info->context, CL_MEM_READ_WRITE, nalloc * sizeof(int), NULL, &cl_error);
        // TO DO: handle errors, check clCreateBuffer flags

        d_atdat->nalloc = nalloc;
        realloced       = true;
    }

    d_atdat->natoms       = natoms;
    d_atdat->natoms_local = nbat->natoms_local;

    /* need to clear GPU f output if realloc happened */
    if (realloced)
    {
        nbnxn_ocl_clear_f(ocl_nb, nalloc);
    }

    ocl_copy_H2D_async(d_atdat->atom_types, nbat->type, 0,
                      natoms*sizeof(int), ls, &(timers->atdat));

    //cu_copy_H2D_async(d_atdat->atom_types, nbat->type,
    //                  natoms*sizeof(*d_atdat->atom_types), ls);

    if (bDoTime)
    {
        //stat = cudaEventRecord(timers->stop_atdat, ls);
        //CU_RET_ERR(stat, "cudaEventRecord failed");
        cl_int cl_error;

        cl_error = clGetEventProfilingInfo(timers->atdat, CL_PROFILING_COMMAND_START,
            sizeof(cl_ulong), &(timers->start_atdat), NULL);

        cl_error = clGetEventProfilingInfo(timers->atdat, CL_PROFILING_COMMAND_END,
            sizeof(cl_ulong), &(timers->stop_atdat), NULL);
    }
}
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

void free_kernels(cl_kernel *kernels, int count)
{
    int i;    

    for (i = 0; i < count; i++)
        free_kernel(kernels + i);
}

void nbnxn_ocl_free(nbnxn_opencl_ptr_t ocl_nb)
{
    // TO DO: Implement this functions for OpenCL
    int kernel_count;

    /* Free kernels */
    kernel_count = sizeof(ocl_nb->kernel_ener_noprune_ptr) / sizeof(ocl_nb->kernel_ener_noprune_ptr[0][0]);
    free_kernels((cl_kernel*)ocl_nb->kernel_ener_noprune_ptr, kernel_count);

    kernel_count = sizeof(ocl_nb->kernel_ener_prune_ptr) / sizeof(ocl_nb->kernel_ener_prune_ptr[0][0]);
    free_kernels((cl_kernel*)ocl_nb->kernel_ener_prune_ptr, kernel_count);

    kernel_count = sizeof(ocl_nb->kernel_noener_noprune_ptr) / sizeof(ocl_nb->kernel_noener_noprune_ptr[0][0]);
    free_kernels((cl_kernel*)ocl_nb->kernel_noener_noprune_ptr, kernel_count);

    kernel_count = sizeof(ocl_nb->kernel_noener_prune_ptr) / sizeof(ocl_nb->kernel_noener_prune_ptr[0][0]);
    free_kernels((cl_kernel*)ocl_nb->kernel_noener_prune_ptr, kernel_count);

    free_kernel(&(ocl_nb->kernel_memset_f));
    free_kernel(&(ocl_nb->kernel_memset_f2));
    free_kernel(&(ocl_nb->kernel_memset_f3));
    free_kernel(&(ocl_nb->kernel_zero_e_fshift));
}

////void nbnxn_cuda_free(nbnxn_cuda_ptr_t cu_nb)
////{
////    cudaError_t      stat;
////    cu_atomdata_t   *atdat;
////    cu_nbparam_t    *nbparam;
////    cu_plist_t      *plist, *plist_nl;
////    cu_timers_t     *timers;
////
////    if (cu_nb == NULL)
////    {
////        return;
////    }
////
////    atdat       = cu_nb->atdat;
////    nbparam     = cu_nb->nbparam;
////    plist       = cu_nb->plist[eintLocal];
////    plist_nl    = cu_nb->plist[eintNonlocal];
////    timers      = cu_nb->timers;
////
////    if (nbparam->eeltype == eelCuEWALD_TAB || nbparam->eeltype == eelCuEWALD_TAB_TWIN)
////    {
////
////#ifdef TEXOBJ_SUPPORTED
////        /* Only device CC >= 3.0 (Kepler and later) support texture objects */
////        if (cu_nb->dev_info->prop.major >= 3)
////        {
////            stat = cudaDestroyTextureObject(nbparam->coulomb_tab_texobj);
////            CU_RET_ERR(stat, "cudaDestroyTextureObject on coulomb_tab_texobj failed");
////        }
////        else
////#endif
////        {
////            stat = cudaUnbindTexture(nbnxn_cuda_get_coulomb_tab_texref());
////            CU_RET_ERR(stat, "cudaUnbindTexture on coulomb_tab_texref failed");
////        }
////        cu_free_buffered(nbparam->coulomb_tab, &nbparam->coulomb_tab_size);
////    }
////
////    stat = cudaEventDestroy(cu_nb->nonlocal_done);
////    CU_RET_ERR(stat, "cudaEventDestroy failed on timers->nonlocal_done");
////    stat = cudaEventDestroy(cu_nb->misc_ops_done);
////    CU_RET_ERR(stat, "cudaEventDestroy failed on timers->misc_ops_done");
////
////    if (cu_nb->bDoTime)
////    {
////        stat = cudaEventDestroy(timers->start_atdat);
////        CU_RET_ERR(stat, "cudaEventDestroy failed on timers->start_atdat");
////        stat = cudaEventDestroy(timers->stop_atdat);
////        CU_RET_ERR(stat, "cudaEventDestroy failed on timers->stop_atdat");
////
////        /* The non-local counters/stream (second in the array) are needed only with DD. */
////        for (int i = 0; i <= (cu_nb->bUseTwoStreams ? 1 : 0); i++)
////        {
////            stat = cudaEventDestroy(timers->start_nb_k[i]);
////            CU_RET_ERR(stat, "cudaEventDestroy failed on timers->start_nb_k");
////            stat = cudaEventDestroy(timers->stop_nb_k[i]);
////            CU_RET_ERR(stat, "cudaEventDestroy failed on timers->stop_nb_k");
////
////            stat = cudaEventDestroy(timers->start_pl_h2d[i]);
////            CU_RET_ERR(stat, "cudaEventDestroy failed on timers->start_pl_h2d");
////            stat = cudaEventDestroy(timers->stop_pl_h2d[i]);
////            CU_RET_ERR(stat, "cudaEventDestroy failed on timers->stop_pl_h2d");
////
////            stat = cudaStreamDestroy(cu_nb->stream[i]);
////            CU_RET_ERR(stat, "cudaStreamDestroy failed on stream");
////
////            stat = cudaEventDestroy(timers->start_nb_h2d[i]);
////            CU_RET_ERR(stat, "cudaEventDestroy failed on timers->start_nb_h2d");
////            stat = cudaEventDestroy(timers->stop_nb_h2d[i]);
////            CU_RET_ERR(stat, "cudaEventDestroy failed on timers->stop_nb_h2d");
////
////            stat = cudaEventDestroy(timers->start_nb_d2h[i]);
////            CU_RET_ERR(stat, "cudaEventDestroy failed on timers->start_nb_d2h");
////            stat = cudaEventDestroy(timers->stop_nb_d2h[i]);
////            CU_RET_ERR(stat, "cudaEventDestroy failed on timers->stop_nb_d2h");
////        }
////    }
////
////#ifdef TEXOBJ_SUPPORTED
////    /* Only device CC >= 3.0 (Kepler and later) support texture objects */
////    if (cu_nb->dev_info->prop.major >= 3)
////    {
////        stat = cudaDestroyTextureObject(nbparam->nbfp_texobj);
////        CU_RET_ERR(stat, "cudaDestroyTextureObject on nbfp_texobj failed");
////    }
////    else
////#endif
////    {
////        stat = cudaUnbindTexture(nbnxn_cuda_get_nbfp_texref());
////        CU_RET_ERR(stat, "cudaUnbindTexture on nbfp_texref failed");
////    }
////    cu_free_buffered(nbparam->nbfp);
////
////    if (nbparam->vdwtype == evdwCuEWALDGEOM || nbparam->vdwtype == evdwCuEWALDLB)
////    {
////#ifdef TEXOBJ_SUPPORTED
////        /* Only device CC >= 3.0 (Kepler and later) support texture objects */
////        if (cu_nb->dev_info->prop.major >= 3)
////        {
////            stat = cudaDestroyTextureObject(nbparam->nbfp_comb_texobj);
////            CU_RET_ERR(stat, "cudaDestroyTextureObject on nbfp_comb_texobj failed");
////        }
////        else
////#endif
////        {
////            stat = cudaUnbindTexture(nbnxn_cuda_get_nbfp_comb_texref());
////            CU_RET_ERR(stat, "cudaUnbindTexture on nbfp_comb_texref failed");
////        }
////        cu_free_buffered(nbparam->nbfp_comb);
////    }
////
////    stat = cudaFree(atdat->shift_vec);
////    CU_RET_ERR(stat, "cudaFree failed on atdat->shift_vec");
////    stat = cudaFree(atdat->fshift);
////    CU_RET_ERR(stat, "cudaFree failed on atdat->fshift");
////
////    stat = cudaFree(atdat->e_lj);
////    CU_RET_ERR(stat, "cudaFree failed on atdat->e_lj");
////    stat = cudaFree(atdat->e_el);
////    CU_RET_ERR(stat, "cudaFree failed on atdat->e_el");
////
////    cu_free_buffered(atdat->f, &atdat->natoms, &atdat->nalloc);
////    cu_free_buffered(atdat->xq);
////    cu_free_buffered(atdat->atom_types, &atdat->ntypes);
////
////    cu_free_buffered(plist->sci, &plist->nsci, &plist->sci_nalloc);
////    cu_free_buffered(plist->cj4, &plist->ncj4, &plist->cj4_nalloc);
////    cu_free_buffered(plist->excl, &plist->nexcl, &plist->excl_nalloc);
////    if (cu_nb->bUseTwoStreams)
////    {
////        cu_free_buffered(plist_nl->sci, &plist_nl->nsci, &plist_nl->sci_nalloc);
////        cu_free_buffered(plist_nl->cj4, &plist_nl->ncj4, &plist_nl->cj4_nalloc);
////        cu_free_buffered(plist_nl->excl, &plist_nl->nexcl, &plist->excl_nalloc);
////    }
////
////    sfree(atdat);
////    sfree(nbparam);
////    sfree(plist);
////    if (cu_nb->bUseTwoStreams)
////    {
////        sfree(plist_nl);
////    }
////    sfree(timers);
////    sfree(cu_nb->timings);
////    sfree(cu_nb);
////
////    if (debug)
////    {
////        fprintf(debug, "Cleaned up CUDA data structures.\n");
////    }
////}
////
////void cu_synchstream_atdat(nbnxn_cuda_ptr_t cu_nb, int iloc)
////{
////    cudaError_t  stat;
////    cudaStream_t stream = cu_nb->stream[iloc];
////
////    stat = cudaStreamWaitEvent(stream, cu_nb->timers->stop_atdat, 0);
////    CU_RET_ERR(stat, "cudaStreamWaitEvent failed");
////}

//wallclock_gpu_t * nbnxn_cuda_get_timings(nbnxn_cuda_ptr_t cu_nb)
//{
//    return (cu_nb != NULL && cu_nb->bDoTime) ? cu_nb->timings : NULL;
//}

//void nbnxn_cuda_reset_timings(nonbonded_verlet_t* nbv)
//{
//    if (nbv->cu_nbv && nbv->cu_nbv->bDoTime)
//    {
//        init_timings(nbv->cu_nbv->timings);
//    }
//}

//int nbnxn_cuda_min_ci_balanced(nbnxn_cuda_ptr_t cu_nb)
//{
//    return cu_nb != NULL ?
//           gpu_min_ci_balanced_factor*cu_nb->dev_info->prop.multiProcessorCount : 0;
//
//}


int nbnxn_ocl_min_ci_balanced(nbnxn_opencl_ptr_t ocl_nb)
{
    return ocl_nb != NULL ?
           gpu_min_ci_balanced_factor * ocl_nb->dev_info->compute_units : 0;     
}

gmx_bool nbnxn_ocl_is_kernel_ewald_analytical(const nbnxn_opencl_ptr_t ocl_nb)
{
    return ((ocl_nb->nbparam->eeltype == eelOclEWALD_ANA) ||
            (ocl_nb->nbparam->eeltype == eelOclEWALD_ANA_TWIN));
}
