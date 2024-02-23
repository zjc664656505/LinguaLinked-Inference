In order to compile model splitting, minizinc needs to be installed.
1. On linux, use "snap install minizinc --classic" to install minizinc to your machine.
2. After minizinc is installed, please save your constraint processing model as model.mzn and your data as data.dzn.
3. Then use default solver of minizinc "Gecode" to solve your model using your data "minizinc --solver model.mzn data.dzn"
