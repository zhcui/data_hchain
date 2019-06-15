#! /usr/bin/env python
import numpy as np
import scipy as sp
from   scipy.io import FortranFile
from   scipy    import linalg as LA
la = LA
import sys, os

from libdmet_solid.system.integral import Integral, dumpFCIDUMP_no_perm
from libdmet_solid.utils.misc import mdot
from libdmet_solid.solver.dmrgci import split_localize, gaopt, reorder, cas_from_1pdm

def read1pdm(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    nsites = int(lines[0])
    pdm = np.zeros((nsites, nsites))

    for line in lines[1:]:
        tokens = line.split()
        pdm[int(tokens[0]), int(tokens[1])] = float(tokens[2])

    return pdm
def onepdm(norb, restricted, bogoliubov=False, tmpDir='.'):
    if restricted:
        #rho = read1pdm(os.path.join(tmpDir, "/node0/spatial_onepdm.0.0.txt")) / 2
        rho = read1pdm(os.path.join(tmpDir, "spatial_onepdm.0.0.txt")) / 2
        rho = rho.reshape((1, norb, norb))
    else:
        #rho0 = read1pdm(os.path.join(tmpDir, "/node0/onepdm.0.0.txt"))
        rho0 = read1pdm(os.path.join(tmpDir, "onepdm.0.0.txt"))
        rho = np.empty((2, norb, norb))
        rho[0] = rho0[::2, ::2]
        rho[1] = rho0[1::2, 1::2]
    if bogoliubov:
        #kappa = read1pdm(os.path.join(tmpDir, "/node0/spatial_pairmat.0.0.txt"))
        kappa = read1pdm(os.path.join(tmpDir, "spatial_pairmat.0.0.txt"))
        if restricted:
            kappa = (kappa + kappa.T) / 2
        GRho = np.zeros((norb*2, norb*2))
        GRho[:norb, :norb] = rho[0]
        GRho[norb:, :norb] = -kappa.T
        GRho[:norb, norb:] = -kappa
        if restricted:
            GRho[norb:, norb:] = np.eye(norb) - rho[0]
        else:
            GRho[norb:, norb:] = np.eye(norb) - rho[1]
        return GRho
    else:
        return rho





#-----------------------------------------------#

def read_one_body_gms():
 fi    = open("one_body_gms","r")
 fl    = fi.readlines()
 nb,nl = [int(x) for x in fl[0].split()]
 S1    = np.zeros((nb,nb))
 H1    = np.zeros((nb,nb))
 for il in range(2,nl+2):
  ir,ic,sr,si = [float(x) for x in fl[il].split()]
  ir,ic       = int(ir),int(ic)
  assert(np.abs(si)<1e-6)
  S1[ir-1,ic-1] = sr
  S1[ic-1,ir-1] = sr
 for il in range(nl+4,2*nl+4):
  ir,ic,hr,hi = [float(x) for x in fl[il].split()]
  ir,ic       = int(ir),int(ic)
  assert(np.abs(hi)<1e-6)
  H1[ir-1,ic-1] = hr
  H1[ic-1,ir-1] = hr
 return nb,S1,H1

#-----------------------------------------------#

def read_two_body_gms(nb):
 f = FortranFile('V2b_AO_cholesky.mat','r')
 nd = f.read_record(dtype=np.int32)
 ng = f.read_record(dtype=np.int32)[1]
 L = np.zeros((nb,nb,ng))
 for ig in range(ng):
  X = f.read_reals(np.float)
  L[:,:,ig] = X.reshape((nb,nb))
 #L = L[:,:,:1]
 return 2.0*np.tensordot(L,L,axes=((-1),(-1))).transpose((0,1,3,2)),L 
#2.0*np.einsum('prg,sqg->prqs',L,L),L

# -----------------------------------------------#

def get_eri_mo(nb, L, C):
    from pyscf.lib import einsum
    if C.ndim == 2:
        L_mo = einsum("pP, pqL -> PqL", C, L)
        L_mo = einsum("PqL, qQ -> PQL", L_mo, C)
        eri = einsum("PRG, SQG -> PRQS", L_mo, L_mo) * 2.0
    else:
        eri = np.zeros((3, nb, nb, nb, nb))
        L_a = einsum("pP, pqL -> PqL", C[0], L)
        L_a = einsum("PqL, qQ -> PQL", L_a, C[0])
        L_b = einsum("pP, pqL -> PqL", C[1], L)
        L_b = einsum("PqL, qQ -> PQL", L_b, C[1])
        eri[0] = einsum("PRG, SQG -> PRQS", L_a, L_a) * 2.0
        eri[1] = einsum("PRG, SQG -> PRQS", L_b, L_b) * 2.0
        eri[2] = einsum("PRG, SQG -> PRQS", L_a, L_b) * 2.0
    return eri

#-----------------------------------------------#

def read_LM():
 f     = open('LM_gms','r')
 fl    = f.readlines()
 nb,nl = [int(x) for x in fl[0].split()]
 X = np.zeros((nb,nb),dtype=complex)
 B = np.zeros((nb,nb),dtype=complex)
 for il in range(2,nl+2):
  ir,ic,xr,xi  = [float(x) for x in fl[il].split()]
  ir,ic        = int(ir),int(ic)
  X[ir-1,ic-1] = xr+1j*xi
 for il in range(nl+4,2*nl+4):
  ir,ic,br,bi  = [float(x) for x in fl[il].split()]
  ir,ic        = int(ir),int(ic)
  B[ir-1,ic-1] = br+1j*bi
 return X,B

def check_eri_symmetries(eri):
 print "4-fold?"
 print "   ",np.abs(eri-eri.transpose(2,3,0,1)).max()
 print "   ",np.abs(eri-eri.transpose(1,0,3,2)).max()
 print "   ",np.abs(eri-eri.transpose(3,2,1,0)).max()
 print "8-fold?"
 print "   ",np.abs(eri-eri.transpose(1,0,2,3)).max()
 print "   ",np.abs(eri-eri.transpose(3,2,0,1)).max()
 print "   ",np.abs(eri-eri.transpose(0,1,3,2)).max()
 print "   ",np.abs(eri-eri.transpose(2,3,1,0)).max()

#-----------------------------------------------#

def read_eigen_gms():
 fc    = open('eigen_gms_HF','r')
 fl    = fc.readlines()
 nb,nu = [int(x) for x in fl[0].split()]
 C0 = np.zeros((nb,nu))
 jl = 0
 for il in range(2,nu*(nb+1)+2):
  if(len(fl[il].split())>0):
   cr,ci = [float(x) for x in fl[il].split()]
   ib    = jl%nb
   ip    = jl//nb
   C0[ib,ip] = cr
   jl  += 1
 nb,nd = [int(x) for x in fl[nu*(nb+1)+2].split()]
 C1 = np.zeros((nb,nu))
 jl = 0
 for il in range(nu*(nb+1)+4,(nu+nd)*(nb+1)+4):
  if(len(fl[il].split())>0):
   cr,ci = [float(x) for x in fl[il].split()]
   ib    = jl%nb
   ip    = jl//nb
   C1[ib,ip] = cr
   jl  += 1
 #print np.abs(np.dot(C0.T,C0)-np.eye(nu)).max()
 #print np.abs(np.dot(C1.T,C1)-np.eye(nd)).max()
 return nu,nd,C0,C1

#-----------------------------------------------#

def reconstruct_energy(E0,H1,eri,rho):
 e1  =       np.einsum('pr,rp',H1,rho[0]+rho[1])
 e2d =   0.5*np.einsum('prqs,pr,qs',eri,rho[0]+rho[1],rho[0]+rho[1])
 e2x = -(0.5*np.einsum('prqs,ps,qr',eri,rho[0],rho[0])+0.5*np.einsum('prqs,ps,qr',eri,rho[1],rho[1]))
 print "======================================================"
 print "          reconstructed MF variational energy         "
 print "total        ",E0+e1+e2d+e2x
 print "offset       ",E0
 print "1b           ",e1
 print "2b, direct   ",e2d
 print "2b, exchange ",e2x
 print "======================================================"

#-----------------------------------------------#

def do_HF(nu,nd,H0,S1,H1,eri,rho,X,B):
 import numpy
 import numpy.random
 from   pyscf import gto,scf,ao2mo 
 import pyscf.scf.stability
 import pyscf.scf.uhf
 import types

 mol = gto.M()
 mol.nelectron     = nu+nd
 mol.spin          = nu-nd
 mol.verbose       = 4
 mol.incore_anyway = True

 Emin = 1e4

 for itry in range(1):
  mf            = scf.UHF(mol)
  mf.max_cycle = 0
  mf.energy_nuc = lambda *args: H0
  mf.get_hcore  = lambda *args: H1
  mf.get_ovlp   = lambda *args: S1
  mf._eri       = eri #ao2mo.restore(4,eri ,nb)
  def get_jk_loc(mol=None,dm=None,hermi=None):
   vj = np.zeros((2,dm[0].shape[0],dm[0].shape[1]))
   vk = np.zeros((2,dm[0].shape[0],dm[0].shape[1]))
   for js in range(2):
    vj[js,:,:] = np.tensordot(eri,dm[js],axes=((0,1),(1,0)))
    vk[js,:,:] = np.tensordot(eri,dm[js],axes=((1,2),(0,1)))
#    vj[js,:,:] = np.einsum('pqrs,qp->rs',eri,dm[js])
#    vk[js,:,:] = np.einsum('pqrs,qr->ps',eri,dm[js])
   return vj,vk
  mf.get_jk = get_jk_loc
  mf.diis_space       = 20
  mf.diis_start_cycle = 1
  #E = mf.kernel([rho[0]+0.1*np.random.random((nb,nb)),rho[1]+0.1*np.random.random((nb,nb))])
  E = mf.kernel([rho[0],rho[1]])
  #print "stability analysis"
  #stab = mf.stability()
  #if(not mf.converged):
  if False:
   #print "not converged, Newton phase"
   mf  = mf.newton()
   rdm = mf.make_rdm1()
   E   = mf.kernel(rdm)
   #print "stability analysis"
   #stab = mf.stability()
  rdm = mf.make_rdm1()
  C   = mf.mo_coeff
 
  if(E<Emin):
   mfmin = mf
   Emin  = E
   Rmin  = rdm
   Cmin  = C

 print "\n *********************** \n"
 print "minimum hf energy ",Emin
# print "stability analysis"
# mfmin.stability()
# print "HF energies"
# for ee in mf.mo_energy:
#  print ee
# Cl   = [C[0][:,:nu],C[1][:,:nd]]
# XX   = LA.expm(2*np.pi*1j*X/(nu+nd)*1.1)
# num  = LA.det( np.dot(Cl[0][:,:nu].T,Cl[0][:,:nu]) )
# num *= LA.det( np.dot(Cl[1][:,:nd].T,Cl[1][:,:nd]) )
# dnA  = LA.det( np.dot(Cl[0][:,:nu].T,np.dot(XX,Cl[0][:,:nu])) )
# dnA *= LA.det( np.dot(Cl[1][:,:nd].T,np.dot(XX,Cl[1][:,:nd])) )
# dnB  = LA.det( np.dot(Cl[0][:,:nu].T,np.dot( B,Cl[0][:,:nu])) )
# dnB *= LA.det( np.dot(Cl[1][:,:nd].T,np.dot( B,Cl[1][:,:nd])) )
# print "localization, strategy A ",np.abs(dnA/num)
# print "localization, strategy B ",np.abs(dnB/num)
# print "\n *********************** \n"
# 
# from pyscf import cc
# mycc = cc.UCCSD(mfmin)
# if(mol.spin==0): mycc.level_shift      = 0.15
# mycc.kernel()
# from pyscf.cc import uccsd_t
# uccsd_t.kernel(mycc,mycc.ao2mo(mf.mo_coeff))
#
# if(not mycc.converged):
#  if(mol.spin==0): mycc.level_shift      = 0.15
#  mycc.kernel(t1=mycc.t1,t2=mycc.t2)
#  from pyscf.cc import uccsd_t
#  uccsd_t.kernel(mycc,mycc.ao2mo(mf.mo_coeff))

 return Emin,Rmin,Cmin

#-----------------------------------------------#

def print_gms(C0,C1):
 fout=open('eigen_gms_HF','w')
 fout.write('%s %s \n\n' % (nb,nu))
 for ip in range(nu):
  for ib in range(nb):
   fout.write('%f 0.000000\n' % (C0[ib,ip]))
  fout.write('\n')
 fout.write('%s %s \n\n' % (nb,nd))
 for ip in range(nd):
  for ib in range(nb):
   fout.write('%f 0.000000\n' % (C1[ib,ip]))
  fout.write('\n')

def get_offset(f):
 import re
 import sys
 for line in open(f,'r'):
  if re.search('Subtotal',   line): E1 = float(line.split()[2])
  if re.search('Variational',line): E2 = float(line.split()[3])
 return E2-E1

def get_Ham_LMO(norb, nu, nd, H0, H1, L, C, restricted=False, bogoliubov=False):
    #from libdmet.utils import logger as log
    #log.verbose = "DEBUG2"
    
    # get Ham in MO basis
    h1_mo = np.asarray([mdot(C[s].conj().T, H1, C[s]) for s in range(2)])
    eri_mo = get_eri_mo(norb, L, C)
    Ham = Integral(norb, restricted, bogoliubov, H0, {"cd": h1_mo}, {"ccdd": eri_mo})
    
    # (_ncore, _npart, _nvirt)
    info = [(nu, 0, norb-nu), (nd, 0, norb-nd)]
    casHam, cas, rotmat = split_localize(C, info, Ham, basis = None) 
    #order = gaopt(casHam, tmp = "./")
    #log.info("Orbital order: %s", order)
    # reorder casHam and cas
    #casHam, cas = reorder(order, casHam, cas)
    return casHam

def get_Ham_LMO_from_1pdm(norb, nu, nd, H0, H1, L, rho, restricted=True, bogoliubov=False):
    from libdmet_solid.utils import logger as log
    log.verbose = "DEBUG2"
    
    ncas = norb
    nelecas = nu + nd
    nelec = nelecas
    core, cas, virt, info = cas_from_1pdm(rdm1_res, ncas, nelecas, nelec)
    C = np.asarray([cas])
    info = (info,)
    # get Ham in MO basis
    if restricted:
        h1_mo = np.asarray([mdot(C[s].conj().T, H1, C[s]) for s in range(1)])
        eri_mo = get_eri_mo(norb, L, C[0])[np.newaxis]
    else:
        h1_mo = np.asarray([mdot(C[s].conj().T, H1, C[s]) for s in range(2)])
        eri_mo = get_eri_mo(norb, L, C)
    

    Ham = Integral(norb, restricted, bogoliubov, H0, {"cd": h1_mo}, {"ccdd": eri_mo})
    
    # (_ncore, _npart, _nvirt)
    #info = [(nu, 0, norb-nu), (nd, 0, norb-nd)]
    casHam, cas, rotmat = split_localize(C, info, Ham, basis = None) 
    np.save('cas_no.npy', cas)
    np.save('rotmat_no.npy', rotmat)
    #order = gaopt(casHam, tmp = "./")
    #log.info("Orbital order: %s", order)
    # reorder casHam and cas
    #casHam, cas = reorder(order, casHam, cas)
    return casHam


#-----------------------------------------------#
np.set_printoptions(3, linewidth=1000, suppress=True)
nb,S1,H1    = read_one_body_gms()
#X,B         = read_LM()
eri,L       = read_two_body_gms(nb)
nu,nd,C0,C1 = read_eigen_gms()

#H0    = get_offset('INFO-0')
H0    = get_offset('INFO-0.25')
nu    = 2
nd    = 2
rho   = [ np.dot(C0[:,:nu],C0[:,:nu].T) , np.dot(C1[:,:nd],C1[:,:nd].T) ]

norb = nb
nocc = np.asarray([nu, nd])
C = np.asarray([C0, C1])


rdm_uhf = np.asarray([C[s][:, :nocc[s]].dot(C[s][:, :nocc[s]].conj().T) for s in range(2)])


rdm1_ao = rdm_uhf

rdm1_res = (rdm1_ao[0]+rdm1_ao[1]) * 0.5

Ham = get_Ham_LMO_from_1pdm(norb, nu, nd, H0, H1, L, rdm1_res, restricted=True)

print "Ham complete"

dumpFCIDUMP_no_perm("./FCIDUMP", Ham)

exit()
