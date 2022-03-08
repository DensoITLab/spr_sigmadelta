# #######################################################################
# Event-Generator
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def getEvent(x0, x_lft, th=0):
    if len(x_lft)==0:
        x_lft = 0*x0
    do = x0 - x_lft
    if th>0:
        delta = do.hardshrink(th)
        epsilon = do - delta
        x_lft = x0 - epsilon
    else:
        delta = do
        x_lft = x0
    return delta, x_lft