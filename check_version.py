import sklearn
import sys
import inspect
from sklearn.metrics import mean_squared_error

print(f"--- Environment Diagnosis ---")
print(f"Python Executable: {sys.executable}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Scikit-learn path: {sklearn.__file__}")
print(f"mean_squared_error signature: {inspect.signature(mean_squared_error)}")
print(f"---------------------------")
