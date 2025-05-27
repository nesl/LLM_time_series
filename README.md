# Can LLMs Classify Time Series Data?
> 📑 [Report](./report.md)  

### Abstract
The increasing prevalence of time-series data across domains, such as healthcare, finance, and IoT, has driven the need for flexible and generalizable modeling approaches. Traditional time-series analysis relies heavily on supervised models, which suffer from data scarcity and high deployment costs. This project explores the potential of Large Language Models (LLMs) as general-purpose few-shot learners for time-series classification tasks. Specifically, we investigate how different input representations affect LLM performance in two representative tasks: Dual-Tone Multi-Frequency (DTMF) signal decoding and Human Activity Recognition (HAR). By comparing text-based, visual, and multimodal inputs across models including GPT-4o, GPT-o1, and DeepSeek-R1, we assess their reasoning capabilities and robustness. Our findings show that while LLMs demonstrate potential, performance significantly varies depending on input representation and model type. In DTMF, GPT-o1 and DeepSeek-R1 consistently outperforms GPT-4o, particularly in tasks requiring text-based numerical reasoning. In HAR, visualization aids interpretation, and few-shot learning significantly boosts performance. However, challenges remain, especially in precise plot reading, domain knowledge retrieval (GPT-4o), and multimodal integration. Further domain-specific enhancements and robust representation strategies are heavily required for current LLMs.

# DTMF
## 1. Run the experiments:
`python dtmf_run.py <subcommand> -m <model> -n <noise-type> [optional arguments]`

### Required Arguments
1. -m, --model:  
    Choose the model (LLM) for this task. Support models: GPT-4o, GPT-o1, DppeSeek-R1
    - Options: gpt-4o, o1-2024-12-17, deepseek-reasoner
1. -n, --noise-type:  
    Choose input data noise type. 'clean' means using data generated with exactly DTMF frequencies, 'noise' means using data generated with added noise.
    - Options: noise, clean

### Optional Arguments
1. -r, --result-save-filename:  
    The file name to save results. (default: `results_{model}_{noise-type}_{subcommand}[_{optional arguments}].csv`)

### Subcommands (Input Types)
1. **freq_text**: Raw frequency-magnitude text input  
    `python dtmf_run.py freq_text -m <model> -n <noise-type> [-g]`  
    Optional Arguments:  
    1. -g, --guide:  
        Add step-by-step guidance if setted, otherwise no guidance will be added in the prompt
1. **freq_plot**: Plot frequency-magnitude pairs into line plot for input  
    `python dtmf_run.py freq_plot -m <model> -n <noise-type> [-g -gr]`  
    Note that *freq_plot* only supports model GPT-4o and GPT-o1.  
    Optional Arguments:  
    1. -g, --guide:  
        Add step-by-step guidance if setted, otherwise no guidance will be added in the prompt
    1. -gr, --grid:  
        Add grid lines to the input plots is setted, otherwise input plots will not contain any grid lines
1. **freq_pair**: Input low/high frequency pair directly  
    `python dtmf_run.py freq_pair -m <model> -n <noise-type> [-map]`  
    Optional Arguments:  
    1. -map, --map:  
        Provide the true DTMF frequency-key mapping in the map if setted.


## 2. Evaluation:
`python dtmf_eval.py -r <result-save-filename> [optional arguments]`  

### Required Argument:  
1. -r, --result-save-filename:  
    Results file name used in `dtmf_run.py`. A prefix of "result_" and a file extension of ".csv" will be automatically added.  

### Optional Arguments:
1. -e, --err-tolerance:  
    An integer of error tolerance range for frequency detection (default: 15 for *freq_plot* results, 5 for results of all other input types)
1. --no-detail-acc:  
    Calculate step-by-step accuracies or not. If guidance is included, step-by-step accuracies will be calculated by default, set --no-detail-acc to **disable** this feature. Otherwise only overall accuracy will be calculated, regardless of whether this argument is setted or not.


# Human Activity Recognition (HAR)
## 1. Run the experiments:
`python shl_run.py -m <model> -i <input_type> [optional arguments]`

### Required Arguments
1. -m, --model:  
    Choose the model (LLM) for this task. Support models: GPT-4o, GPT-o1, DppeSeek-R1
    - Options: gpt-4o, o1-2024-12-17, deepseek-reasoner
1. -i, --input:  
    Select the input representation format.
	- Options:
	    - time_text: IMU time-series as raw text
	    - time_text_fewshot: Raw text + one raw text example for each class
	    - time_text_description: Raw text + textual summary  
        *(Note: the following input types only suppport model GPT-4o and GPT-o1.)*  
	    - time_plot: Time-series line plot
	    - time_plot_fewshot: Line plot + one line plot example for each class
	    - time_plot_env: Line plot + environment photo taken when the activity is happening
	    - env_only: Environment photo only

### Optional Arguments
1. -dn, --data-num:  
    Number of test samples per class (default: 30)
1. -df, --data-folder:  
    Path to the dataset folder (default: ./datasets/SHL_processed/User1/220617/Torso_video/)
1. -l, --location:  
    The body location on which the smartphone used for IMU data collection is placed (default: Torso)
1. -f, --frequency:  
    Sampling frequency in Hz (default: 10 for *time_text_fewshot* and *time_text_description*, otherwise 100)
1. -r, --result-save-filename:  
    The file name to save results. Will be default to `results_{model}_User1_220617_{4*data-num}_{location}_{input}_4class.csv` if not setted.

## 2. Evaluation:
`python shl_eval.py -r <result-save-filename>`  

### Required Argument:  
1. -r, --result-save-filename:  
    Results file name used in `shl_run.py`. A prefix of "result_" and a file extension of ".csv" will be automatically added. This filename will also be used to generate the confusion matrix file name.  