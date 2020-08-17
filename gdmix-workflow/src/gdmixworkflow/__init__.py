# DO NOT COPY & PASTE THIS CODE!!!!!
#
# This is a special file only needed for "src/gdmixworkflow/__init__.py"
# to declare the "gdmixworkflow" package as a "namespace"
#
# All other "__init__.py" files can just be blank, or contain normal Python
# module code.
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)
