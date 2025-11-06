"""
Provides a wrapper class that facilitates the usage of the
General Boolean Function Benchmark Suite (GBFS):

https://dl.acm.org/doi/abs/10.1145/3594805.3607131
"""

import os
from sklearn.base import RegressorMixin
from pathlib import Path
from src.benchmark.benchmark import Benchmark
from src.gp.functions import AND, OR, BUFA, NOTA, NOR, NAND, XOR, XNOR, NOT
from src.benchmark.logic_synthesis.ls_benchmark import LSBenchmark
from src.benchmark.logic_synthesis.boolean_benchmark_tools.benchmark_evaluator import BenchmarkEvaluator
from src.gp.problem import BlackBox
import src.gp.util as util
import numpy as np

strfun = {
    "AND": AND,
    "OR": OR,
    "BUFA": BUFA,
    "NOTA": NOTA,
    "NOT": NOT,
    "NAND": NAND,
    "NOR": NOR,
    "XOR": XOR,
    "XNOR": XNOR
}


class LSBench(Benchmark):
    benchmarks: dict
    data_dir: str

    def __init__(self, data_dir_: str):
        self.benchmarks = {}
        self.data_dir = data_dir_
        self.generate()

    def generate(self, args: any = None):
        self.benchmarks["add3"] = LSBenchmark(file_=self.data_dir + "/tt/add3.tt", name_="add3")
        self.benchmarks["add4"] = LSBenchmark(file_=self.data_dir + "/tt/add4.tt", name_="add4")
        self.benchmarks["add5"] = LSBenchmark(file_=self.data_dir + "/tt/add5.tt", name_="add7")
        self.benchmarks["add6"] = LSBenchmark(file_=self.data_dir + "/tt/add6.tt", name_="add6")
        self.benchmarks["add7"] = LSBenchmark(file_=self.data_dir + "/tt/add7.tt", name_="add7")
        self.benchmarks["add8"] = LSBenchmark(file_=self.data_dir + "/tt/add8.tt", name_="add8")

        self.benchmarks["mul3"] = LSBenchmark(file_=self.data_dir + "/tt/mul3.tt", name_="mul3")
        self.benchmarks["mul4"] = LSBenchmark(file_=self.data_dir + "/tt/mul4.tt", name_="mul4")
        self.benchmarks["mul5"] = LSBenchmark(file_=self.data_dir + "/tt/mul5.tt", name_="mul5")

        self.benchmarks["epar8"] = LSBenchmark(file_=self.data_dir + "/tt/epar8.tt", name_="epar8")
        self.benchmarks["epar9"] = LSBenchmark(file_=self.data_dir + "/tt/epar9.tt", name_="epar9")
        self.benchmarks["epar10"] = LSBenchmark(file_=self.data_dir + "/tt/epar10.tt", name_="epar10")
        self.benchmarks["epar11"] = LSBenchmark(file_=self.data_dir + "/tt/epar11.tt", name_="epar11")

        self.benchmarks["icomp5"] = LSBenchmark(file_=self.data_dir + "/tt/icomp5.tt", name_="icomp5")
        self.benchmarks["icomp6"] = LSBenchmark(file_=self.data_dir + "/tt/icomp6.tt", name_="icomp6")
        self.benchmarks["icomp7"] = LSBenchmark(file_=self.data_dir + "/tt/icomp7.tt", name_="icomp7")
        self.benchmarks["icomp8"] = LSBenchmark(file_=self.data_dir + "/tt/icomp8.tt", name_="icomp8")
        self.benchmarks["icomp9"] = LSBenchmark(file_=self.data_dir + "/tt/icomp9.tt", name_="icomp9")

        self.benchmarks["mcomp3"] = LSBenchmark(file_=self.data_dir + "/tt/mcomp3.tt", name_="mcomp3")
        self.benchmarks["mcomp4"] = LSBenchmark(file_=self.data_dir + "/tt/mcomp4.tt", name_="mcomp4")
        self.benchmarks["mcomp5"] = LSBenchmark(file_=self.data_dir + "/tt/mcomp5.tt", name_="mcomp5")
        self.benchmarks["mcomp6"] = LSBenchmark(file_=self.data_dir + "/tt/mcomp6.tt", name_="mcomp6")

        self.benchmarks["alu3"] = LSBenchmark(file_=self.data_dir + "/tt/alu3.tt", name_="alu3")
        self.benchmarks["alu4"] = LSBenchmark(file_=self.data_dir + "/tt/alu4.tt", name_="alu4")
        self.benchmarks["alu5"] = LSBenchmark(file_=self.data_dir + "/tt/alu5.tt", name_="alu5")
        self.benchmarks["alu6"] = LSBenchmark(file_=self.data_dir + "/tt/alu6.tt", name_="alu6")
        self.benchmarks["alu7"] = LSBenchmark(file_=self.data_dir + "/tt/alu7.tt", name_="alu7")
        self.benchmarks["alu8"] = LSBenchmark(file_=self.data_dir + "/tt/alu8.tt", name_="alu8")

        self.benchmarks["enc8"] = LSBenchmark(file_=self.data_dir + "/tt/onehot_enc8.tt", name_="enc8")
        self.benchmarks["enc16"] = LSBenchmark(file_=self.data_dir + "/tt/onehot_enc16.tt", name_="enc16")
        self.benchmarks["enc32"] = LSBenchmark(file_=self.data_dir + "/tt/onehot_enc32.tt", name_="enc32")

        self.benchmarks["dec4"] = LSBenchmark(file_=self.data_dir + "/tt/onehot_dec4.tt", name_="dec4")
        self.benchmarks["dec8"] = LSBenchmark(file_=self.data_dir + "/tt/onehot_dec8.tt", name_="dec8")
        self.benchmarks["dec16"] = LSBenchmark(file_=self.data_dir + "/tt/onehot_dec16.tt", name_="dec16")

        self.benchmarks["count4"] = LSBenchmark(file_=self.data_dir + "/tt/onescount4.tt", name_="count4")
        self.benchmarks["count6"] = LSBenchmark(file_=self.data_dir + "/tt/onescount6.tt", name_="count6")
        self.benchmarks["count8"] = LSBenchmark(file_=self.data_dir + "/tt/onescount8.tt", name_="count8")
        self.benchmarks["count10"] = LSBenchmark(file_=self.data_dir + "/tt/onescount10.tt", name_="count10")

    def add3(self):
        return self.benchmarks["add3"]

    def add4(self):
        return self.benchmarks["add4"]

    def add5(self):
        return self.benchmarks["add5"]

    def add6(self):
        return self.benchmarks["add6"]

    def add7(self):
        return self.benchmarks["add7"]

    def add8(self):
        return self.benchmarks["add8"]

    def mul3(self):
        return self.benchmarks["mul3"]

    def mul4(self):
        return self.benchmarks["mul4"]

    def mul5(self):
        return self.benchmarks["mul5"]

    def epar8(self):
        return self.benchmarks["epar8"]

    def epar9(self):
        return self.benchmarks["epar9"]

    def epar10(self):
        return self.benchmarks["epar10"]

    def epar11(self):
        return self.benchmarks["epar11"]

    def icomp3(self):
        return self.benchmarks["icomp3"]

    def icomp4(self):
        return self.benchmarks["icomp4"]

    def icomp5(self):
        return self.benchmarks["icomp5"]

    def icomp6(self):
        return self.benchmarks["icomp6"]

    def icomp7(self):
        return self.benchmarks["icomp6"]

    def icomp8(self):
        return self.benchmarks["icomp7"]

    def icomp9(self):
        return self.benchmarks["icomp8"]

    def mcomp3(self):
        return self.benchmarks["mcomp3"]

    def mcomp4(self):
        return self.benchmarks["mcomp4"]

    def mcomp5(self):
        return self.benchmarks["mcomp5"]

    def mcomp6(self):
        return self.benchmarks["mcomp6"]

    def alu3(self):
        return self.benchmarks["alu3"]

    def alu4(self):
        return self.benchmarks["alu4"]

    def alu5(self):
        return self.benchmarks["alu5"]

    def alu6(self):
        return self.benchmarks["alu6"]

    def alu7(self):
        return self.benchmarks["alu7"]

    def alu8(self):
        return self.benchmarks["alu8"]

    def enc8(self):
        return self.benchmarks["enc8"]

    def enc16(self):
        return self.benchmarks["enc16"]

    def enc32(self):
        return self.benchmarks["enc32"]

    def dec4(self):
        return self.benchmarks["dec4"]

    def dec8(self):
        return self.benchmarks["dec8"]

    def dec16(self):
        return self.benchmarks["dec16"]

    def count4(self):
        return self.benchmarks["count4"]

    def count6(self):
        return self.benchmarks["count6"]

    def count8(self):
        return self.benchmarks["count8"]

    def count10(self):
        return self.benchmarks["count10"]

    class LSRegressor(RegressorMixin):

        def __init__(self,
                     representation_,
                     config_,
                     hyperparameters_,
                     functions_=["AND", "OR", "BUFA", "NOT"],
                     terminals_=None):

            if terminals_ is None:
                terminals_ = []
            self.program = None
            self.model = None
            self.representation = representation_
            self.config = config_
            self.hyperparameters = hyperparameters_
            self.functions = [strfun[f] for f in functions_]
            self.terminals = terminals_
            self.evaluator = BenchmarkEvaluator()
            self.loss = self.evaluator.hamming_distance
            self.fitted_ = False

        def fit(self, X, y, checkpoint=None):
            problem = BlackBox(X, y, self.loss, ideal_ = 0, minimizing_ = True)

            self.model = util.get_model(self.representation, self.functions, self.terminals,
                                        self.hyperparameters, self.config)

            if checkpoint is not None:
                self.model.resume(checkpoint, problem)

            self.program = self.model.evolve(problem)
            self.fitted_ = True

        def predict(self, X):
            if not self.fitted_:
                raise ValueError("Model not fitted")
            return np.array([self.model.predict(self.program.genome, x)[0] for x in X])
