# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from ai_models.model import Model

LOG = logging.getLogger(__name__)


class FengWu(Model):
    expver = "fgwu"

    download_url = "https://get.ecmwf.int/repository/test-data/ai-models/fengwu/{file}"
    download_files = [
        "data_mean.npy",
        "data_std.npy",
        "fengwu_v1.onnx",
        "fengwu_v2.onnx",
    ]

    area = [90, 0, -90, 360]
    grid = [0.25, 0.25]
    param_sfc = ["10u", "10v", "2t", "msl"]
    param_level_pl = (
        ["z", "q", "u", "v", "t"],
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    )

    def __init__(self, num_threads=1, **kwargs):
        super().__init__(**kwargs)
        self.num_threads = num_threads
        self.hour_steps = 6

    def load_model(self):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = self.num_threads

        model_file = os.path.join(self.assets, f"fengwu_v{self.model_version}.onnx")
        os.stat(model_file)
        with self.timer(f"Loading {model_file}"):
            model = ort.InferenceSession(
                model_file,
                sess_options=options,
                providers=self.providers,
            )
        return model

    def get_init_time(self):
        init_time = self.all_fields.order_by(valid_datetime="descending")[0].datetime()
        init_time = pd.to_datetime(init_time)
        return init_time

    def create_input(self, init_time):
        init_time_str = init_time.strftime("%Y-%m-%dT%H:%M:%S")
        hist_time = init_time - pd.Timedelta(hours=6)
        hist_time_str = hist_time.strftime("%Y-%m-%dT%H:%M:%S")

        param_sfc = self.param_sfc
        param_pl, level = self.param_level_pl
        fields_pl = self.fields_pl
        fields_sfc = self.fields_sfc

        fields_pl_init = fields_pl.sel(valid_datetime=init_time_str, param=param_pl, level=level)
        fields_pl_init = fields_pl_init.order_by(param=param_pl, level=level)
        pl_init = defaultdict(list)
        for field in fields_pl_init:
            pl_init[field.metadata("param")].append(field)
        fields_sfc_init = fields_sfc.sel(valid_datetime=init_time_str, param=param_sfc)
        fields_sfc_init = fields_sfc_init.order_by(param=param_sfc)
        sfc_init = defaultdict(list)
        for field in fields_sfc_init:
            sfc_init[field.metadata("param")].append(field)
        input_init = []
        for param, fields in sfc_init.items():
            data = np.stack([field.to_numpy(dtype=np.float32) for field in fields]).reshape(-1, 1, 721, 1440)
            input_init.append(data)
            info = f"name: {param}, shape: {data.shape}, range: {data.min():.3f} ~ {data.max():.3f}"
            LOG.info(info)
        for param, fields in pl_init.items():
            data = np.stack([field.to_numpy(dtype=np.float32) for field in fields]).reshape(-1, 13, 721, 1440)
            input_init.append(data)
            info = f"name: {param}, shape: {data.shape}, range: {data.min():.3f} ~ {data.max():.3f}"
            LOG.info(info)
        input_init = np.concatenate(input_init, axis=1)

        fields_pl_hist = fields_pl.sel(valid_datetime=hist_time_str, param=param_pl, level=level)
        fields_pl_hist = fields_pl_hist.order_by(param=param_pl, level=level)
        pl_hist = defaultdict(list)
        for field in fields_pl_hist:
            pl_hist[field.metadata("param")].append(field)
        fields_sfc_hist = fields_sfc.sel(valid_datetime=hist_time_str, param=param_sfc)
        fields_sfc_hist = fields_sfc_hist.order_by(param=param_sfc)
        sfc_hist = defaultdict(list)
        for field in fields_sfc_hist:
            sfc_hist[field.metadata("param")].append(field)
        input_hist = []
        for param, fields in sfc_hist.items():
            data = np.stack([field.to_numpy(dtype=np.float32) for field in fields]).reshape(-1, 1, 721, 1440)
            input_hist.append(data)
            info = f"name: {param}, shape: {data.shape}, range: {data.min():.3f} ~ {data.max():.3f}"
            LOG.info(info)
        for param, fields in pl_hist.items():
            data = np.stack([field.to_numpy(dtype=np.float32) for field in fields]).reshape(-1, 13, 721, 1440)
            input_hist.append(data)
            info = f"name: {param}, shape: {data.shape}, range: {data.min():.3f} ~ {data.max():.3f}"
            LOG.info(info)
        input_hist = np.concatenate(input_hist, axis=1)

        self.template_pl = fields_pl_init.sel(valid_datetime=init_time_str)
        self.template_sfc = fields_sfc_init.sel(valid_datetime=init_time_str)

        return input_hist, input_init

    def parse_model_args(self, args):
        res_dict = {}
        for i in range(0, len(args), 2):
            res_dict[args[i]] = args[i + 1]
        self.model_version = res_dict.get("--version", 1)
        LOG.info(f"Use model version {self.model_version}")

    def run(self):
        # create input
        data_mean_file = os.path.join(self.assets, "data_mean.npy")
        os.stat(data_mean_file)
        data_mean = np.load(data_mean_file)[:, np.newaxis, np.newaxis]
        data_std_file = os.path.join(self.assets, "data_std.npy")
        os.stat(data_std_file)
        data_std = np.load(data_std_file)[:, np.newaxis, np.newaxis]
        init_time = self.get_init_time()
        input_hist, input_init = self.create_input(init_time)
        input_hist_after_norm = (input_hist.reshape(69, 721, 1440) - data_mean) / data_std
        input_init_after_norm = (input_init.reshape(69, 721, 1440) - data_mean) / data_std
        input = np.concatenate((input_hist_after_norm, input_init_after_norm), axis=0)[np.newaxis, :, :, :]
        input = input.astype(np.float32)

        # load model
        model = self.load_model()

        # run the model
        total_step = self.lead_time // self.hour_steps
        with self.stepper(6) as stepper:
            for i in range(total_step):
                step = (i + 1) * self.hour_steps

                (output,) = model.run(None, {"input": input})

                input = np.concatenate((input[:, 69:], output[:, :69]), axis=1)
                output = (output[0, :69] * data_std) + data_mean

                sfc_chans = len(self.param_sfc)
                pl_data = output[sfc_chans:]
                for data, f in zip(pl_data, self.template_pl):
                    self.write(data, template=f, step=step)
                sfc_data = output[:sfc_chans]
                for data, f in zip(sfc_data, self.template_sfc):
                    self.write(data, template=f, step=step)

                stepper(i, step)
