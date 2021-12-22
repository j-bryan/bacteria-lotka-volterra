import numpy as np
from utils.combinations import comb_wr, num_comb_wr


# ==================================
# ===   DATA LIBRARY FUNCTIONS   ===
# ==================================

def gen_lib(X, poly_deg=0, trig_deg=0, exp_deg=0):
    """
    Generates a library of functions to be used in a sparse regression method. May include polynomial, trigonometric,
    exponential, and logarithmic functions. For now, only positive degrees for each type of function are supported.

    :param X: data matrix
    :param poly_deg: max polynomial degree to include (x^deg + x^(deg-1) + ... + C)
    :param trig_deg: max trig degree to include (sin(x) + sin(2x) + ... + sin(deg*x), similar for cos())
    :param exp_deg: max exponential degree to include (e^x + e^2x + ... + e^(deg*x))
    :return library: library of functions Theta(X)
    """
    # check inputs
    assert isinstance(X, np.ndarray)
    assert isinstance(poly_deg, int) and poly_deg >= 0
    assert isinstance(trig_deg, int) and trig_deg >= 0
    assert isinstance(exp_deg, int) and exp_deg >= 0

    X_rows, X_cols = X.shape

    # calculate how many columns the final matrix will have so a new numpy array won't have to be allocated every time
    # you want to add a column or function library
    const_cols = 1
    poly_cols = 0
    for i in range(1, poly_deg + 1):
        poly_cols += num_comb_wr(X_cols, i)
    trig_cols = 2 * trig_deg * X_cols
    exp_cols = exp_deg * X_cols
    cols = const_cols + poly_cols + trig_cols + exp_cols

    # initialize final return array with a constants column
    final = np.ones((X_rows, cols))

    start = 1
    # add polynomial function matrices to final
    if poly_deg != 0:
        for i in range(1, poly_deg + 1):
            stop = start + num_comb_wr(X.shape[1], i)
            final[:, start:stop] = poly_lib(X, deg=i)
            start = stop

    # add trig functions to final
    if trig_deg != 0:
        for i in range(1, trig_deg + 1):
            stop = start + 2 * X_cols
            final[:, start:stop] = trig_lib(X, i)
            start = stop

    # add exponential function columns to matrix
    if exp_deg != 0:
        for i in range(1, exp_deg + 1):
            stop = start + X_cols
            final[:, start:stop] = exp_lib(X, i)
            start = stop

    return final


def poly_lib(X, deg):
    """
    Builds a function library of polynomial function of degree deg, using n different variables.

    :param X: data matrix
    :param deg: polynomial degree
    :return: callable polynomial library
    """
    X_rows, X_cols = X.shape

    # get all combinations of variables for deg degree polynomials
    combs = comb_wr(X_cols, deg)

    # initialize final matrix to return
    Theta = np.ones((X_rows, combs.shape[0]))

    for i in range(combs.shape[0]):  # column in final/row in combs
        for col in combs[i]:
            Theta[:, i] *= X[:, col]

    return Theta


def trig_lib(X, deg):
    return np.hstack((np.sin(deg * X), np.cos(deg * X)))


def exp_lib(X, deg):
    return np.exp(deg * X)


# ======================================
# ===   SYMBOLIC LIBRARY FUNCTIONS   ===
# ======================================

def gen_callable_model_X(Xi, poly_deg=0, trig_deg=0, exp_deg=0):
    """ Creates a callable function from the Xi matrix generated during sparse regression. """
    Theta = gen_callable_lib_X(len(Xi[0]), poly_deg, trig_deg, exp_deg)

    def model(x):
        return np.dot(Theta(x), Xi)

    return model


def gen_callable_lib_X(num_vars, poly_deg=0, trig_deg=0, exp_deg=0):
    """ Creates a callable Theta array for use in gen_callable_model() """
    # validate inputs
    assert num_vars > 0

    poly = callable_poly_lib_X(num_vars, poly_deg)
    trig = callable_trig_lib_X(num_vars, trig_deg)
    exp = callable_exp_lib_X(num_vars, exp_deg)

    def Theta_callable_X(x):
        shape = len(x) if len(x.shape) == 1 else x.shape[0]
        ret = np.ones((shape, 1))

        if poly_deg != 0:
            ret = np.hstack((ret, poly(x)))
        if trig_deg != 0:
            ret = np.hstack((ret, trig(x)))
        if exp_deg != 0:
            ret = np.hstack((ret, exp(x)))
        return ret

    return Theta_callable_X


def callable_poly_lib_X(num_var, deg):
    combs = []
    for i in range(1, deg + 1):
        for comb in comb_wr(num_var, i):
            combs.append(comb)

    def f(x):
        """ :param x: matrix of column vectors x1, x2, etc """
        assert len(x.shape) == 2
        arr = np.ones((x.shape[0], len(combs)))
        for i, c in enumerate(combs):
            for term in c:
                arr[:, i] *= x[:, term]
        return arr

    return f


def callable_trig_lib_X(num_var, deg):
    def f(x):
        arr = np.empty((x.shape[0], 2 * num_var * deg))
        for i in range(0, 2 * num_var * deg - num_var, 2):
            arr[:, i:i+num_var] = np.sin((i + 2) // 2 * x)
            arr[:, i+num_var:i+2*num_var] = np.cos((i + 2) // 2 * x)
        return arr
    return f


def callable_exp_lib_X(num_var, deg):
    def f(x):
        arr = np.empty((x.shape[0], num_var * deg))
        for i in range(0, num_var * deg - 1):
            arr[:, i:i+num_var] = np.exp(i * x)
        return arr
    return f


# ===================================
# ===   FOR USE WITH RK METHODS   ===
# ===================================

def gen_callable_model(Xi, poly_deg=0, trig_deg=0, exp_deg=0):
    num_var = Xi.shape[1]
    lib = [lambda x, t: [1], gen_poly_lib(num_var, poly_deg), gen_trig_lib(num_var, trig_deg), gen_exp_lib(num_var, exp_deg)]
    lib = np.array([f for f in lib if f is not None])

    def model(x, t):
        Theta = []
        for f in lib:
            to_append = f(x, t)
            for xi in to_append:
                Theta.append(xi)
        return np.array(Theta).dot(Xi)

    return model


def gen_poly_lib(num_var, deg):
    if deg == 0:
        return None

    combs = []
    for i in range(1, deg + 1):
        for comb in comb_wr(num_var, i):
            combs.append(comb)

    def f(x, t=0):
        arr = np.ones(len(combs))
        for i, c in enumerate(combs):
            for term in c:
                arr[i] *= x[term]
        return arr

    return f


def gen_trig_lib(num_var, deg):
    if deg == 0:
        return None


def gen_exp_lib(num_var, deg):
    if deg == 0:
        return None
