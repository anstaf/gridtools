# -*- coding: utf-8 -*-
import sys
import ast
import logging
import warnings

import numpy as np




class FunctorBody (ast.NodeVisitor):
    """
    Represents the Do( ) function of a stencil's functor in AST form.-
    """
    def __init__ (self, nodes, params, symbols):
        """
        Constructs a functor body object using the received node:

            node    an AST-node list representing the body of this functor;
            params  a dict of FunctorParameters of this functor;
            symbols the StecilSymbols of all symbols within the stencil where
                    the functor lives.-
        """
        self.params  = params
        self.symbols = symbols
        #
        # initialize an empty C++ code string
        #
        self.cpp = ''
        try:
            if len (nodes) > 0:
                self.nodes = nodes
        except TypeError:
            warnings.warn ("FunctorBody expects a list of AST nodes.",
                           RuntimeWarning)

    def _sign_operator (self, op):
        """
        Returns the sign of an operation.-
        """
        if isinstance (op, ast.Add) or isinstance (op, ast.UAdd):
            sign = '+'
        elif isinstance (op, ast.Sub) or isinstance (op, ast.USub):
            sign = '-'
        elif isinstance (op, ast.Mult):
            sign = '*'
        elif isinstance (op, ast.Div):
            sign = '/'
        elif isinstance (op, ast.Pow):
            #
            # TODO: translate to a multiplication
            #
            sign = None
            logging.warning ("Cannot translate 'x**y'")
        else:
            sign = None
            logging.warning ("Cannot translate '%s'" % str (op))
        return sign

         
    def generate_code (self, src):
        """
        Generates C++ code from the AST backing this object:

            src     the Python source from which the C++ is generated;
                    this is used to display friendly error messages.-
        """
        for n in self.nodes:
            try:
                code = self.visit (n)
                if code is not None:
                    self.cpp += "%s;\n\t\t" % code

            except Exception as e:
                #
                # preprocess the source code to correctly display the line,
                # because comments are lost in the AST translation
                #
                # FIXME: comment_offset is not correctly calculated
                #
                src_lines      = src.split ('\n')
                comment_offset = 0
                for l in src_lines:
                    if l.strip (' ').startswith ('#'):
                        comment_offset += 1

                correct_lineno = n.lineno + comment_offset
                source_line    = src_lines[correct_lineno].strip (' ')
                raise type(e) ("at line %d:\n\t%s" % (correct_lineno,
                                                      source_line))


    def visit_Assign (self, node):
        """
        Generates code from an Assignment node, i.e., expr = expr.-
        """
        for tgt in node.targets:
            return "%s = %s" % (self.visit (tgt),          # lvalue
                                self.visit (node.value))   # rvalue


    def visit_Attribute (self, node):
        """
        Tries to replace attributes with values from the stencil's symbol table.-
        """
        attr_name = "%s.%s" % (node.value.id,
                               node.attr)
        attr_val = self.symbols[attr_name]
        #
        # do not replace strings or NumPy arrays
        #
        if (isinstance (attr_val, str) or
            isinstance (attr_val, np.ndarray)):
            return attr_name
        else:
            return str (attr_val)


    def visit_AugAssign (self, node):
        """
        Generates code for an operation-assignment node, i.e., expr += expr.-
        """
        sign = self._sign_operator (node.op)
        return "%s %s= %s" % (self.visit (node.target),
                              sign,
                              self.visit (node.value))


    def visit_BinOp (self, node):
        """
        Generates code for a binary operation, e.g., +,-,*, ...
        """
        sign = self._sign_operator (node.op)
        #
        # take care of the parenthesis for correct operation precedence
        #
        operand = []
        for op in [node.left, node.right]:
            if (isinstance (op, ast.Num) or 
                isinstance (op, ast.Name) or
                isinstance (op, ast.Attribute) or
                isinstance (op, ast.Subscript)):
                operand.append ('%s' % self.visit (op))
            else:
                operand.append ('(%s)' % self.visit (op))

        return "%s %s %s" % (operand[0], sign, operand[1])


    def visit_Num (self, node):
        """
        Returns the number in this node.-
        """
        return str(node.n)


    def visit_Subscript (self, node):
        """
        Generates code from Subscript node, i.e., expr[expr].-
        """
        if isinstance (node.slice, ast.Index):
            if isinstance (node.slice.value, ast.BinOp):
                #
                # this subscript has shifting
                #
                if isinstance (node.slice.value.op, ast.Add):
                    indexing = '('
                    for e in node.slice.value.right.elts:
                        if isinstance (e, ast.Num):
                            indexing += "%s," % str (e.n)
                        elif isinstance (e, ast.UnaryOp):
                            indexing += "%s%s," % (self._sign_operator (e.op),
                                                   str (e.operand.n))
                        else:
                            logging.error ("Subscript shifting operation unknown")
                    #
                    # strip the last comma off
                    #
                    indexing = '%s)' % indexing[:-1]
                else:
                    indexing = ''
                    logging.warning ("Subscript shifting only supported with '+'")
            elif isinstance (node.slice.value, ast.Name):
                #
                # TODO understand subscripting over 'get_interior_points'
                #
                if node.slice.value.id == 'p':
                    indexing = '( )'
                else:
                    indexing = ''
                    logging.warning ("Ignoring subscript not using 'p'")
            #
            # check if subscripting any data fields
            #
            if isinstance (node.value, ast.Attribute):
                name = node.value.attr
                value = self.symbols[name]
                if isinstance (value, FunctorParameter):
                    return "dom(%s%s)" % (name, indexing)
            #
            # check if subscripting any functor parameters 
            #
            elif isinstance (node.value, ast.Name):
                name = node.value.id
                value = self.symbols[name]
                #
                # FIXME only ask for the parameters of this functor
                #
                if self.symbols.is_parameter (name):
                    return "dom(%s%s)" % (name, indexing)



class FunctorParameter ( ):
    """
    Represents a parameter of a stencil functor.-
    """
    def __init__ (self, name):
        """
        Creates a new parameter with the received name.-
        """
        self.id     = None
        self.name   = None 
        self.dim    = None
        self.input  = None
        self.output = None
        self.set_name (name)


    def set_name (self, name):
        """
        Sets a new name to this functor parameter.-
        """
        #
        # do not add 'self' as a functor parameter
        #
        if name != 'self':
            self.name = name
            #
            # temporary parameters are not 'input' nor 'output'
            #
            self.input  = self.name.startswith ('in_')
            self.output = self.name.startswith ('out_')
        else:
            self.name = None



class StencilFunctor ( ):
    """
    Represents a functor inside a multi-stage stencil.-
    """
    def __init__ (self, name, node, params, symbols):
        """
        Constructs a new StencilFunctor:

            name    a name to uniquely identify this functor;
            node    the For AST node (see
                    https://docs.python.org/3.4/library/ast.html) of the
                    Python comprehention from which this functor will be built;
            params  a dict of FunctorParameters of this functor;
            symbols the StecilSymbols of all symbols within the stencil where
                    the functor lives.-
        """
        self.name = name
        self.params = params
        self.symbols = symbols
        #
        # the body of the functor is inlined from the 'for' loops
        #
        self.body = None
        #
        # the AST node of the Python function representing this functor
        #
        self.set_ast (node)


    def set_ast (self, node):
        """
        Speficies the AST describing the operations this functor should 
        implement:

            node    a FunctionDef AST node (see
                    https://docs.python.org/3.4/library/ast.html).-
        """
        if isinstance (node, ast.For):
            self.node = node 
        else:
            raise TypeError ("Functor's root AST node should be type ast.For")


    def analyze_params (self, nodes):
        """
        Extracts the parameters from the Python function:

            nodes   a list of AST nodes representing the parameters.-
        """
        for p in nodes:
            #
            # do not add the parameter if it already exists as a symbol
            #
            name = p.arg
            if self.symbols[name] is None:
                par = FunctorParameter (name)
                #
                # the name is None if the parameter was ignored/invalid
                #
                if par.name is not None:
                    par.id = len (self.params)
                    self.params[par.name] = par
            else:
                logging.warning ("Not adding parameter '%s' of functor '%s' because it already exists in the symbol table" % (name, self.name))


    def generate_code (self, src):
        """
        Generates the C++ code of this functor:

            src     the Python source from which the C++ is generated;
                    this is used to display friendly error messages.-
        """
        self.body.generate_code (src)
