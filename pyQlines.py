#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os, time
from lmfit.models import GaussianModel, ExponentialModel
from lmfit import report_fit #minimize, Parameters,
# np.set_printoptions(precision=2)

start = time.clock()

cspeed = 2.99792e5  # unit km/s

Hb = 4861.33
OIIIa = 4959.9
OIIIb = 5006.8

Ha = 6562.82
NIIa = 6548.04
NIIb = 6583.41
SIIa = 6716.44
SIIb = 6730.81

# dat = np.loadtxt('rest_contin_unsub.txt')
# x = dat[:, 0]
# y = dat[:, 1] - 26.5262223

dat = np.loadtxt('rest_continsub.txt')
x = dat[:, 0]
y = dat[:, 1] 


xmin1 = 4750.
xmax1 = 5100.
xmin2 = 6350.
xmax2 = 6800.

y1 = y[((x > xmin1) & (x < xmax1)) | ((x > xmin2) & (x < xmax2))]
x1 = x[((x > xmin1) & (x < xmax1)) | ((x > xmin2) & (x < xmax2))]


# exp = ExponentialModel(prefix='exp_')
# pars = exp.guess(y1, x=x1)


offset = 2000. / 2.35 / cspeed * OIIIb

n6716 = GaussianModel(prefix='n6716_')
pars = n6716.guess(y1, x=x1)
pars.update(n6716.make_params())
pars['n6716_center'].set(6716)
pars['n6716_sigma'].set(600. / 2.35 / cspeed * 6716, min=200. /
                     2.35 / cspeed * 6716,  max=800. / 2.35 / cspeed * 6716)
pars['n6716_amplitude'].set(100, min=0)

n6731 = GaussianModel(prefix='n6731_')
pars.update(n6731.make_params())
pars['n6731_center'].set(expr='n6716_center/6716.*6731.')
pars['n6731_sigma'].set(expr='n6716_sigma/6716.*6731.')
pars['n6731_amplitude'].set(100, min=0)

n6583 = GaussianModel(prefix='n6583_')
pars.update(n6583.make_params())
pars['n6583_center'].set(expr='n6716_center/6716.*6583.')
pars['n6583_sigma'].set(expr='n6716_sigma/6716.*6583.')
pars['n6583_amplitude'].set(100, min=0)

n6548 = GaussianModel(prefix='n6548_')
pars.update(n6548.make_params())
pars['n6548_center'].set(expr='n6716_center/6716.*6548.')
pars['n6548_sigma'].set(expr='n6716_sigma/6716.*6548.')
pars['n6548_amplitude'].set(expr='n6583_amplitude/3.')

n6563 = GaussianModel(prefix='n6563_')
pars.update(n6563.make_params())
pars['n6563_center'].set(expr='n6716_center/6716.*6562.2')
pars['n6563_sigma'].set(expr='n6716_sigma/6716.*6562.2')
pars['n6563_amplitude'].set(300, min=0)

b6563 = GaussianModel(prefix='b6563_')
pars.update(b6563.make_params())
pars['b6563_center'].set(Ha, min=Ha - offset, max=Ha + offset)
pars['b6563_sigma'].set(5000. / 2.35 / cspeed * Ha, min=4000. /
                     2.35 / cspeed * Ha,  max=6000. / 2.35 / cspeed * Ha)
pars['b6563_amplitude'].set(8000, min=0)

a6563 = GaussianModel(prefix='a6563_')
pars.update(a6563.make_params())
pars['a6563_center'].set(Ha, min= Ha - 3, max=Ha + 3)
pars['a6563_sigma'].set(300. / 2.35 / cspeed * Ha, min=100. /
                     2.35 / cspeed * Ha,  max=500. / 2.35 / cspeed * Ha)
pars['a6563_amplitude'].set(-10, max=0)

n5007 = GaussianModel(prefix='n5007_')
pars.update(n5007.make_params())
pars['n5007_center'].set(OIIIb, min=OIIIb - offset, max=OIIIb + offset)
pars['n5007_sigma'].set(800. / 2.35 / cspeed * OIIIb, min=400. /
                     2.35 / cspeed * OIIIb,  max=1200. / 2.35 / cspeed * OIIIb)
pars['n5007_amplitude'].set(1200)

b5007 = GaussianModel(prefix='b5007_')
pars.update(b5007.make_params())
pars['b5007_center'].set(OIIIb, min=OIIIb - offset, max=OIIIb)
pars['b5007_sigma'].set(1800. / 2.35 / cspeed * OIIIb,
                     max=2400. / 2.35 / cspeed * OIIIb)
pars['b5007_amplitude'].set(500)

n4959 = GaussianModel(prefix='n4959_')
pars.update(n4959.make_params())
pars['n4959_center'].set(expr='n5007_center/5007.*4959.')
pars['n4959_sigma'].set(expr='n5007_sigma/5007.*4959.')
pars['n4959_amplitude'].set(expr='n5007_amplitude/3.')

b4959 = GaussianModel(prefix='b4959_')
pars.update(b4959.make_params())
pars['b4959_center'].set(expr='b5007_center/5007.*4959.')
pars['b4959_sigma'].set(expr='b5007_sigma/5007.*4959.')
pars['b4959_amplitude'].set(expr='b5007_amplitude/3.')

n4861 = GaussianModel(prefix='n4861_')
pars.update(n4861.make_params())
pars['n4861_center'].set(expr='n6716_center/6716.*4861.')
pars['n4861_sigma'].set(expr='n6716_sigma/6716.*4861.')
pars['n4861_amplitude'].set(expr='n6563_amplitude/b6563_amplitude*b4861_amplitude')

b4861 = GaussianModel(prefix='b4861_')
pars.update(b4861.make_params())
pars['b4861_center'].set(Hb, min=Hb - offset, max=Hb + offset)
pars['b4861_sigma'].set(expr='b6563_sigma/6563.*4861.')
pars['b4861_amplitude'].set(1200, min=0)

a4861 = GaussianModel(prefix='a4861_')
pars.update(a4861.make_params())
pars['a4861_center'].set(Hb, min= Hb - 3, max=Hb + 3)
pars['a4861_sigma'].set(expr='a6563_sigma/6563.*4861.')
pars['a4861_amplitude'].set(expr='a6563_amplitude/b6563_amplitude*b4861_amplitude')

mod = n5007 + n4959 +\
	n4861 + b4861 + a4861 +\
	n6563 + b6563 + a6563 +\
	n6548 + n6583 + n6716 + n6731

init = mod.eval(pars, x=x1)
out = mod.fit(y1, pars, x=x1)
comps = out.eval_components(x=x1)

print(out.fit_report(min_correl=0.5))

plt.plot(x1, y1 			,'k-'	,linewidth=0.5)
# plt.plot(x1, init, 'k--')
# plt.plot(x1, comps['exp_']		,'r--'	,linewidth=0.5	, alpha=0.5, label='Exp')
plt.plot(x1, comps['n5007_']	,'g-'	,linewidth=0.5	, alpha=0.5, label='n_OIII5007')
# plt.plot(x1, comps['b5007_']	,'g--'	,linewidth=0.5	, alpha=0.5, label='b_OIII5007')
plt.plot(x1, comps['n4959_']	,'b-'	,linewidth=0.5	, alpha=0.5, label='n_OIII4959')
# plt.plot(x1, comps['b4959_']	,'b--'	,linewidth=0.5	, alpha=0.5, label='b_OIII4959')
plt.plot(x1, comps['n4861_']	,'m'	,linewidth=0.5	, alpha=0.5, label='n_Hb4861')
plt.plot(x1, comps['b4861_']	,'m--'	,linewidth=0.5	, alpha=0.5, label='b_Hb4861')
plt.plot(x1, comps['a4861_']	,'c-'	,linewidth=0.5	, alpha=0.5, label='a_Hb4861')

plt.plot(x1, out.best_fit	,'r-'	,linewidth=0.5	, alpha=1)
plt.xlabel('rest-frame wavelenght [$\AA$]')
plt.ylabel('flux [$10^{-17} erg\ cm^{-2}s^{-1}\AA^{-1}$]')
plt.legend(loc='upper left')
plt.xlim(xmin1, xmax1)
plt.ylim(-40, 120)
plt.savefig("fitting4861.png", dpi=300)

plt.plot(x1, comps['n6563_']	,'m'	,linewidth=0.5	, alpha=0.5, label='n_Ha6563')
plt.plot(x1, comps['b6563_']	,'m--'	,linewidth=0.5	, alpha=0.5, label='b_Ha6563')
plt.plot(x1, comps['a6563_']	,'c-'	,linewidth=0.5	, alpha=0.5, label='a_Ha6563')

plt.plot(x1, comps['n6548_']	,'g-'	,linewidth=0.5	, alpha=0.5, label='n_6548')
plt.plot(x1, comps['n6583_']	,'b-'	,linewidth=0.5	, alpha=0.5, label='n_6583')
plt.plot(x1, comps['n6716_']	,'y-'	,linewidth=0.5	, alpha=0.5, label='n_6716')
plt.plot(x1, comps['n6731_']	,'b-'	,linewidth=0.5	, alpha=0.5, label='n_6731')

# plt.plot(x1, comps['b6548_']	,'g--'	,linewidth=0.5	, alpha=0.5, label='b_6548')
# plt.plot(x1, comps['b6583_']	,'b--'	,linewidth=0.5	, alpha=0.5, label='b_6583')
# plt.plot(x1, comps['b6716_']	,'g--'	,linewidth=0.5	, alpha=0.5, label='b_6716')
# plt.plot(x1, comps['b6731_']	,'b--'	,linewidth=0.5	, alpha=0.5, label='b_6731')

plt.xlim(xmin2, xmax2)
plt.ylim(-40, 120)
# plt.legend(loc='upper left')
plt.savefig("fitting6563.png", dpi=300)


os.system('open fitting*.png')

elapsed = (time.clock() - start)
print('Time used:', '%0.4f' % elapsed, 'seconds.')


