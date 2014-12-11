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
/*! \file
 *  \brief Declare infrastructure for OpenCL JIT compilation for Gromacs
 *
 *  \author Dimitrios Karkoulis <dimitris.karkoulis@gmail.com>
 *  \author Anca Hamuraru <anca@streamcomputing.eu>
 *  \author Teemu Virolainen <teemu@streamcomputing.eu>
 *
 * TODO Currently this file handles compilation of NBNXN kernels,
 * but e.g. organizing the defines for various physics models
 * is leaking in here a bit.
 */

#ifndef OCL_COMPILER_H
#define OCL_COMPILER_H

#include "gromacs/legacyheaders/types/hw_info.h"
#include "../ocl_tools/oclutils.h"

/**
 * \brief Vendor specific kernel sources
 * 
 * Only affects the bottom level kernel sources (nbnxn_ocl_kernel_[spec].cl)
 */
typedef enum{
    _generic_vendor_kernels_ = 0, /**< Standard (warp-less) source file with generated methods/energy/prune */
    _nvidia_vendor_kernels_     , /**< Nvidia source file with generated methods/energy/prune */
    _amd_vendor_kernels_        , /**< AMD source file with generated methods/energy/prune */
    _auto_vendor_kernels_         /**< Compiler will select source based on vendor id*/
} kernel_vendor_spec_t;

/**
 * \brief Kernel sources index
 * 
 * For now there is only default source. One may add here future kernel versions etc.
 * This affect the top level kernel sources (nbnxn_ocl_kernels.cl)
 */
typedef enum{
    _default_source_ = 0  /* The default top-level source  */
} kernel_source_index_t;

/**
 * \brief Full description of the requested Gromacs algorithm flavour
 */
typedef struct
{
    int eeltype;  /**< Value must be of enum eelOcl type */
    int vdwtype;  /**< Value must be of enum evdwOcl type */
} gmx_algo_family_t;


cl_int
ocl_compile_program(
    kernel_source_index_t       kernel_source_file,
    kernel_vendor_spec_t        kernel_vendor_spec,
    gmx_algo_family_t *         p_gmx_algo_family,
    int                         DoFastGen,
    char *                      result_str,
    cl_context                  context,
    cl_device_id                device_id,
    ocl_vendor_id_t             ocl_device_vendor,
    cl_program *                p_program
);

#endif /* OCL_COMPILER_H */
