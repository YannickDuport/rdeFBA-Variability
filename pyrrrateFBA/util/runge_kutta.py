"""
RUNGE KUTTA
Collection of Runge-Kutta parameters for some prominent examples
"""
import numpy as np


class RungeKuttaPars:
    """
    Collect the parameters of the Butcher tableau for several Runge-Kutta schemes
    """
    def __init__(self, s=2, family='LobattoIIIA', force_set=None):
        """
        Create the RK matrix and weight vector based on the underlying (collocation) scheme.
        Numbers are hardcoded
        """
        if not isinstance(s, int) or s < 1:
            raise ValueError('Stage number s of RK method must be a positive integer')
        # --------------------------------------------------------------------
        if force_set is None:
            # It might make sense to ask for things like FSAL, zero rows, is_collocation etc.
            self.family = family
            #
            if family == 'LobattoIIIA':
                if s == 1:
                    raise ValueError('Lobatto methods must have at least two stages')
                elif s == 2: # Trapezoidal rule
                    self.A = np.array([[0, 0],
                                       [0.5, 0.5]])
                    self.b = np.array([[0.5, 0.5]])
                elif s == 3:
                    self.A = np.array([[0, 0, 0],
                                       [5/24, 1/3, -1/24],
                                       [1/6, 2/3, 1/6]])
                    self.b = np.array([[1/6, 2/3, 1/6]])
                else:
                    raise ValueError('LobattoIIIA methods only implemented for s < 4')
                    # TODO: For higher orders: Get from simplified conditions(?)
            # --------------------------------------------------------------------
            elif family == 'RadauIIA':
                if s == 1: # implicit Euler
                    self.A = np.array([[1.0]])
                    self.b = np.array([[1.0]])
                elif s == 2: # Radau 3
                    self.A = np.array([[5/12, -1/12],
                                       [3/4, 1/4]])
                    self.b = np.array([[3/4, 1/4]])
                elif s == 3: # RADAU5
                    self.A = np.array([[(88-7*np.sqrt(6))/360,
                                        (296-169*np.sqrt(6))/1800,
                                        (-2+3*np.sqrt(6))/225],
                                       [(296+169*np.sqrt(6))/1800,
                                        (88+7*np.sqrt(6))/360,
                                        (-2-3*np.sqrt(6))/225],
                                       [(16-np.sqrt(6))/36, (16+np.sqrt(6))/36, 1/9]])
                    self.b = np.array([[(16-np.sqrt(6))/36, (16+np.sqrt(6))/36, 1/9]])
                else:
                    raise ValueError('RadauIIA methods only implemented for s < 4')
            # --------------------------------------------------------------------
            elif family == 'Gauss':
                if s == 1: # implicit midpoint rule
                    self.A = np.array([[0.5]])
                    self.b = np.array([[1.0]])
                elif s == 2:
                    self.A = np.array([[1/4, (3-2*np.sqrt(3))/12],
                                       [(3+2*np.sqrt(3))/12, 1/4]])
                    self.b = np.array([[0.5, 0.5]])
                elif s == 3:
                    self.A = np.array([[5/36, (10-3*np.sqrt(15))/45, (25-6*np.sqrt(15))/180],
                                       [(10+3*np.sqrt(15))/72, 2/9, (10-3*np.sqrt(15))/72],
                                       [(25+6*np.sqrt(15))/180, (10+3*np.sqrt(15))/45, 5/36]])
                    self.b = np.array([[5/18, 4/9, 5/18]])
                else:
                    raise ValueError('Gauss methods only implemented for s < 4')
            # --------------------------------------------------------------------
            elif family == 'Explicit1':
                if s == 1: # explicit Euler
                    self.A = np.array([[0.0]])
                    self.b = np.array([[1.0]])
                elif s == 2: # Heun
                    self.A = np.array([[0.0, 0.0],[1.0, 0.0]])
                    self.b = np.array([[0.5, 0.5]])
                elif s == 3: # Kutta/Simpson
                    self.A = np.array([[0.0, 0.0, 0.0],
                                       [1/2, 0.0, 0.0],
                                       [-1.0, 2.0, 0.0]])
                    self.b = np.array([[1/6, 2/3, 1/6]])
                elif s == 4: # RK4
                    self.A = np.array([[0.0, 0.0, 0.0, 0.0],
                                       [0.5, 0.0, 0.0, 0.0],
                                       [0.0, 0.5, 0.0, 0.0],
                                       [0.0, 0.0, 1.0,0.0]])
                    self.b = np.array([[1/6, 1/3, 1/3, 1/6]])
                else:
                    raise ValueError('Explicit methods (I) only implemented for s < 5')
            else:
                raise ValueError('Unknown Runge-Kutta family: ', family)
            # Create vector c based on entries of matrix A (this is actually not necessarily
            # true for all RK schemes)
            self.update_c()
        else:  # Just believe me, I know what I am doing...
            self.family = None
            self.A = force_set['A']
            self.b = force_set['b']
            self.c = force_set['c']


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__) # A very generic print


    def get_stage_number(self):
        """
        Stage number s of RK scheme
        """
        return self.b.size

    def update_c(self, c_in=None):
        """
        calculate/set stage vector c
        """
        if c_in is None:
            self.c = np.sum(self.A, axis=1, keepdims=True) # based on knot condition
                                                           # Potentially not always true (RadauIA)
        else:
            self.c = c_in


    #def check_simplified_conditions(self, cond_type):
    #def special_matrices # matrices from simplified conditions -> automatically generate
    #def has_fsal(self):, def has_unnatural_coefficients(self):
        # check for 0 <= c_i,b_i <= 1, sum(b) = 1
    # dense output helpers?

    def adjoint_method(self):
        """
        Create a dict with parameters of the "reversed-time method"
        """
        s_rk = self.get_stage_number()
        b = np.zeros((1, s_rk))
        c = np.zeros((s_rk, 1))
        amat = np.zeros((s_rk, s_rk))
        for i in range(s_rk):
            b[0, i] = self.b[0, s_rk-i-1]
            c[i, 0] = 1.0 - self.c[s_rk-i-1, 0]
            for j in range(s_rk):
                amat[i, j] = self.b[0, s_rk-j-1] - self.A[s_rk-i-1, s_rk-j-1]
        return {'A': amat, 'b': b, 'c': c}


    def check_order_conditions(self, order=1):
        """
        Hard-Coded test of order conditions: Outputs a list which is supposed to contain zeros when
        'order' is lower/equal the order of the method (for c_i = sum a_ij)
        @MAYBE: Mark linear order conditions
        """
        if order == 1:
            return np.sum(self.b) - 1.0
        if order == 2:
            return np.dot(self.b, self.c) - 0.5
        if order == 3:
            return [ \
              (np.dot(self.b, (self.c**2)) - 1/3)[0, 0],
              (np.dot(self.b, np.dot(self.A, self.c)) - 1/6)[0, 0]
              ]
        if order == 4:
            return [ \
             (np.dot(self.b, self.c**3) - 0.25)[0, 0],
             (np.dot(self.b*self.c.T, np.dot(self.A, self.c)) - 0.125)[0, 0],
             (np.dot(self.b, np.dot(self.A, self.c**2)) - 1/12)[0, 0],
             (np.dot(self.b, np.dot(self.A, np.dot(self.A, self.c))) - 1/24)[0, 0]
             ]
        if order == 5:
            return [ \
             (np.dot(self.b, self.c**4) - 0.2)[0, 0],
             (np.dot(self.b*(self.c.T**2), np.dot(self.A, self.c)) - 0.1)[0, 0],
             (np.dot(self.b*self.c.T, np.dot(self.A, self.c**2)) - 1/15)[0, 0],
             (np.dot(self.b*self.c.T, np.dot(self.A, np.dot(self.A, self.c))) - 1/30)[0, 0],
             (np.dot(self.b, np.dot(self.A, self.c)*np.dot(self.A, self.c)) - 1/20)[0, 0],
             (np.dot(self.b, np.dot(self.A, self.c**3)) - 1/20)[0, 0],
             (np.dot(self.b, np.dot(self.A, self.c*np.dot(self.A, self.c))) - 1/40)[0, 0],
             (np.dot(self.b, np.dot(self.A, np.dot(self.A, self.c**2))) - 1/60)[0, 0],
             (np.dot(self.b, np.dot(self.A, np.dot(self.A, np.dot(self.A, self.c)))) - 1/120)[0, 0]
             ]
        raise ValueError('Higher order conditions not yet implemented')



class SimplifiedConditions:
    """
    Simplified Conditions B(s), C(s), D(s) in the sense of Butcher
    """
    def __init__(self, rkm):
        s = rkm.get_stage_number()
        # C_s = (c_i^j/j)
        self.Cs = np.hstack([rkm.c**j/j for j in range(1, s+1)])
        self.eH = np.array([[1/j for j in range(1, s+1)]])
        self.B = np.diag(rkm.b.flatten())
        self.Ns = np.tile(np.array([1/j for j in range(1, s+1)]), (s, 1))
        self.Vs = np.hstack([rkm.c**j for j in range(s)])
