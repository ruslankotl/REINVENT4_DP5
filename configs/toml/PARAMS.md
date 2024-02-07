# TOML parameters

This is a summary of TOML parameters for each run mode.

## Sampling

Sample a number of SMILES with associated NLLs.


| Parameter          | Description                                                                                            |
|--------------------|--------------------------------------------------------------------------------------------------------|
| run\_type          | set to "sampling"                                                                                      |
| use\_cuda          | "true" to use GPU, "false" to use CPU                                                                  |
| json\_out\_config    | filename of the TOML file in JSON format                                                               |
| [parameters]       | starts the parameter section                                                                           |
| model\_file        | filename to model file from which to sample                                                            |
| smiles\_file       | filename for inpurt SMILES for Lib/LinkInvent and Mol2Mol                                              |
| sample\_strategy   | Mol2Mol only: "beamsearch" or "multinomial"                                                          |
| output\_file       | filename for the CSV file with samples SMILES and NLLs                                                 |
| num\_smiles        | number of SMILES to sample, note: this is multiplied by the number of input SMILES                     |
| unique\_molecules  | if "true" only return unique canonicalized SMILES                                                      |
| randomize\_smiles  | if "true" shuffle atoms in input SMILES randomly                                                       |
| tb\_logdir         | if not empty string name of the TensorBoard logging directory                                          |
| temperature        | Mol2Mol only: default 1.0                                                                            |
| target\_smiles\_path | Mol2Mol only: if not empty, filename to provided SMILES, check NLL of generating the provided SMILES |


## Scoring

Interface to the scoring component.  Does not use any models.

| Parameter           | Description                                                                                             |
|---------------------|---------------------------------------------------------------------------------------------------------|
| run\_type           | set to "scoring"                                                                                        |
| use\_cuda           | "true" to use GPU, "false" to use CPU                                                                   |
| json\_out\_config     | filename of the TOML file in JSON format                                                                |
| [parameters]        | starts the parameter section                                                                            |
| smiles\_file        | SMILES filename, SMILES are expected in the first column                                                |
| [scoring\_function] | starts the section for scoring function setup                                                           |
| [[components]]      | start the section for a component within [scoring\_function] , note the double brackets to start a list |
| type                | "custom\_sum" for weighted arithmetic mean or "custom\_produc" for weighted geometric mean                |
| component\_type     | name of the component, FIXME: list all                                                                  |
| name                | a user chosen name for ouput in CSV files, etc.                                                         |
| weight              | the weight for this component                                                                           |


## Transfer Learning

Run transfer learning on a set of input SMILES.

| Parameter              | Description                                                   |
|------------------------|---------------------------------------------------------------|
| run\_type              | set to "transfer\_learning"                                    |
| use\_cuda              | "true" to use GPU, "false" to use CPU                         |
| json\_out\_config        | filename of the TOML file in JSON format                      |
| tb\_logdir             | if not empty string name of the TensorBoard logging directory |
| number_of_cpus         | optional parameter to control number of cpus to generate pairs. if not provided the maximum cpus will be allocated. |
| [parameters]           | starts the parameter section                                  |
| num\_epochs            | number of epochs to run                                       |
| save\_every\_n\_epochs | save checkpoint file every N epochs                           |
| batch\_size            | batch size, note: affects SGD                                 |
| num\_refs              | number of references for similarity                           |
| sample\_batch\_size    | number of samples for similarity                              |
| input\_model\_file     | filename of input prior model                                 |
| smiles\_file           | SMILES file for Lib/Linkinvent and Molformer                  |
| output_model\_file     | filename of the final model                                   |
| pairs.upper\_threshold | Molformer: upper similarity                                   |
| pairs.lower\_threshold | Molformer: lower similarity                                   |
| pairs.min\_cardinality | Molformer:                                                    |
| pairs.max\_cardinality | Molformer:                                                    |


## Staged Learning

Run reinforcement learning (RL) and/or curriculum learning (CL).  CL is simply a multi-stage RL learning.

| Parameter            | Description                                                                                                                                |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| run\_type            | set to "transfer\_learning"                                                                                                                 |
| use\_cuda            | "true" to use GPU, "false" to use CPU                                                                                                      |
| json\_out\_config      | filename of the TOML file in JSON format                                                                                                   |
| tb\_logdir           | if not empty string name of the TensorBoard logging directory                                                                              |
| [parameters]         | starts the parameter section                                                                                                               |
| summary\_csv\_prefix | prefix for output CSV filename                                                                                                             |
| use\_checkpoint      | if "true" use diversity filter from agent\_file if present                                                                                 |
| purge\_memories      | if "true" purge all diversity filter memories (scaffold, SMILES) after each stage |
| prior\_file          | filename of the prior model file, serves as reference                                                                                      |
| agent\_file          | filename of the agent model file, used for training, replace with checkpoint file from previous stage when needed                          |
| batch\_size          | batch size, note: affects SGD                                                                                                              |
| uniquify\_smiles     | if "true" only return unique SMILES (sampling)                                                                                             |
| randomize\_smiles    | if "true" shuffle atoms in input SMILES randomly (sampling)                                                                                |
| [learning\_strategy]  | start section for RL learning strategy                                                                                                     |
| type                 | use "dap"                                                                                                                                  |
| sigma                | sigma in the reward function                                                                                                               |
| rate                 | learning rate for the torch optimizer                                                                                                      |
| [diversity\_filter]   | starts the section for the diversity filter                                                                                                |
| type                 | name of the filter type: "IdenticalMurckoScaffold", "IdenticalTopologicalScaffold", "ScaffoldSimilarity", "PenalizeSameSmiles" |
| bucket\_size         | number of scaffolds to store before molecule is scored zero                                                                                |
| minscore             | minimum score                                                                                                                              |
| minsimilarity        | minimum similarity in "ScaffoldSimilarity"                                                                                                 |
| penalty\_multiplier  | penalty penalty for each molecule in "PenalizeSameSmiles"                                                                                  |
| [inception]          | starts the inception section                                                                                                               |
| smiles\_file         | filename for the "good" SMILES                                                                                                             |
| memory\_size         | number of SMILES to hold in inception memory                                                                                               |
| sample\_size         | number of SMILES randomly sampled from memory                                                                                              |
| [[stage]]            | starts a stage, note the double brackets                                                                                                   |
| chkpt\_file           | filename of the checkpoint file, will be written on termination and Ctrl-C                                                                 |
| termination          | use "simple", termination criterion                                                                                                        |
| max\_score            | maximum score when to terminate                                                                                                            |
| min\_steps            | minimum number of RL steps to avoid early termination                                                                                      |
| max\_steps            | maximum number of RL steps to run, if maximum is hit _all_ stages will be terminated                                                       |

The scoring functions are added as in scoring but prefixed with stage.
