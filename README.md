# transformer-based-fall-detection
Transformer-based Fall Detection in Videos

## Configuration files

There are two configuration files: `config_eval1.yaml` and `config_eval2.yaml`, used for the evaluation strategy 1 and the evaluation strategy 2, respectively. Within them there are several options to tune the experiments. The values in the repository were the ones used to obtain the best results.

## Data

The code expects the UP-Fall dataset format, i.e. 

```
├── Subject 1
│   ├── Activity1
|   |   ├── Trial1
|   |   |   ├── Camera1
|   |   |   |    ├── frame_0000.png
|   |   |   |    ├── frame_0001.png
|   |   ├── Trial 2
|   |   └── Trial 3
│   └── Activity2
├── Subject 2
```

The folder containing this tree structure is declared in the configuration file using the option `dataset_folder`.

## Citation

If you plan to use this code, please, cite us:

```
TODO
```