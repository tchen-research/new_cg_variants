
#include <petsc/private/kspimpl.h>

/*
     KSPSetUp_CHCG - Sets up the workspace needed by the CHCG method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
static PetscErrorCode KSPSetUp_CHCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get work vectors needed by CHCG */
  ierr = KSPSetWorkVecs(ksp,5);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
 KSPSolve_CHCG - This routine actually applies the pipelined conjugate gradient method

 Input Parameter:
 .     ksp - the Krylov space object that was set to use conjugate gradient, by, for
             example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPCG);
*/
static PetscErrorCode  KSPSolve_CHCG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    alpha = 0.0, beta = 0.0, mu = 0.0, delta = 0.0, gamma = 0.0, nu = 0.0, nu_old = 0.0;
  PetscReal      dp    = 0.0;
  Vec            X,B,R,RT,P,S,ST;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;
 
  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  R  = ksp->work[0];
  RT = ksp->work[1];
  P  = ksp->work[2];
  S  = ksp->work[3];
  ST = ksp->work[4];
  
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  /* initialize */
  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);  /*   r <- b - Ax  */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);               /*   r <- b       */
  }
  
  ierr = KSP_PCApply(ksp,R,RT);CHKERRQ(ierr);        /*   rt <- Br     */
  
  ierr = VecCopy(RT,P);CHKERRQ(ierr);                /*   p <- rt      */
  
  ierr = KSP_MatMult(ksp,Amat,P,S);CHKERRQ(ierr);    /*   s  <- A p     */
  ierr = KSP_PCApply(ksp,S,ST);CHKERRQ(ierr);        /*   st <- B s     */
  
  ierr = VecDotBegin(RT,R,&nu);CHKERRQ(ierr);
  ierr = VecDotBegin(P,S,&mu);CHKERRQ(ierr);
  ierr = VecDotBegin(ST,S,&gamma);CHKERRQ(ierr);
  
  ierr = VecDotEnd(RT,R,&nu);CHKERRQ(ierr);          /*   nu    <- (rt,r)  */
  ierr = VecDotEnd(P,S,&mu);CHKERRQ(ierr);           /*   mu    <- (p,s)   */
  ierr = VecDotEnd(ST,S,&gamma);CHKERRQ(ierr);       /*   gamma <- (st,s)  */
  delta = mu;
 
  i = 0;
  do {
    
   /* Compute appropriate norm */  
   switch (ksp->normtype) {
     case KSP_NORM_PRECONDITIONED:
        ierr = VecNormBegin(RT,NORM_2,&dp);CHKERRQ(ierr);
        ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)RT));CHKERRQ(ierr);
        ierr = VecNormEnd(RT,NORM_2,&dp);CHKERRQ(ierr);
        break;
    case KSP_NORM_UNPRECONDITIONED:
        ierr = VecNormBegin(R,NORM_2,&dp);CHKERRQ(ierr);
        ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R));CHKERRQ(ierr);
        ierr = VecNormEnd(R,NORM_2,&dp);CHKERRQ(ierr);
        break;
    case KSP_NORM_NATURAL:
        dp = PetscSqrtReal(PetscAbsScalar(nu));
        break;
    case KSP_NORM_NONE:
        dp   = 0.0;
        break;
    default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
    }

    ksp->rnorm = dp;
    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    /* update scalars */
    alpha = nu/mu;
    nu_old = nu;
    nu = nu_old-2*alpha*delta+(alpha*alpha)*gamma;    
    beta = nu/nu_old;

    /* update vectors */
    ierr = VecAXPY(X, alpha,P);CHKERRQ(ierr);         /*   x  <- x  + alpha * p   */
    ierr = VecAXPY(R,-alpha,S);CHKERRQ(ierr);         /*   r  <- r  - alpha * s   */
    ierr = VecAXPY(RT,-alpha,ST);CHKERRQ(ierr);       /*   rt <- rt - alpha * st  */
    ierr = VecAYPX(P,beta,RT);CHKERRQ(ierr);          /*   p  <- rt + beta  * p   */

    
    
    
    ierr = KSP_MatMult(ksp,Amat,P,S);CHKERRQ(ierr);   /*   s  <- A p              */
    ierr = KSP_PCApply(ksp,S,ST);CHKERRQ(ierr);       /*   st <- B s              */
 
    /* start inner products and matrix products */
    ierr = VecDotBegin(RT,R,&nu);CHKERRQ(ierr);
    ierr = VecDotBegin(P,S,&mu);CHKERRQ(ierr);    
    ierr = VecDotBegin(RT,S,&delta);CHKERRQ(ierr);
    ierr = VecDotBegin(ST,S,&gamma);CHKERRQ(ierr);

  //  ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)S));CHKERRQ(ierr);
    ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R));CHKERRQ(ierr);
    
    ierr = VecDotEnd(RT,R,&nu);CHKERRQ(ierr);         /*   nu    <- (p,s)         */
    ierr = VecDotEnd(P,S,&mu);CHKERRQ(ierr);          /*   mu    <- (p,s)         */
    ierr = VecDotEnd(RT,S,&delta);CHKERRQ(ierr);      /*   delta <- (p,s)         */
    ierr = VecDotEnd(ST,S,&gamma);CHKERRQ(ierr);      /*   gamma <- (p,s)         */


    i++;
    ksp->its = i;

  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}


/*MC
   KSPCHCG - Communication hiding conjugate gradient method.

   This method has only a single non-blocking reduction per iteration, compared to 2 blocking for standard CG. 

   Level: intermediate

   Notes:
   MPI configuration may be necessary for reductions to make asynchronous progress, which is important for performance of pipelined methods.
   See the FAQ on the PETSc website for details.

   Contributed by:
   Tyler Chen, University of Washington, Applied Mathematics Department

   Reference:
   "Pipelined predict-and-recompute conjugate gradient variants". Tyler Chen. In preparation.

   Acknowledgments:
   This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE-1762114. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author and do not necessarily reflect the views of the National Science Foundation.

.seealso: KSPCreate(), KSPSetType(), KSPPIPEPRCG, KSPPIPECG, KSPPIPECR, KSPGROPPCG, KSPPGMRES, KSPCG, KSPCGUseSingleReduction()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_CHCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);

  ksp->ops->setup          = KSPSetUp_CHCG;
  ksp->ops->solve          = KSPSolve_CHCG;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = 0;
  ksp->ops->setfromoptions = 0;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
