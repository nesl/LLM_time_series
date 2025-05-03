import pandas as pd
import textwrap

class PromptConfigDTMF:
    """Configuration class generating prompts for DTMF classification."""
    def __init__(self):
        self.system_prompt = "You are an expertise in dual tone multi-frequency decoding."
    def get_response_format_prompt(self, model, enable_freq=True):
        if model == "deepseek-reasoner":
            if enable_freq:
                return textwrap.dedent("""
                    Please provide your recognized key in the format of <<KEY>> at the first word of your response, followed by a list of high-magnitude frequencies as floating-point numbers rounded to two decimal places, and then followed by your detailed analysis. For example, if you detected the magnitude of frequencies 770Hz, 1477Hz are high and recognized the provided frequency series as key "6", then your response should begin with "<<6>>, [770.00, 1477.00]", followed by deyailed analysis.
                    """)
            else:
                return textwrap.dedent("""
                    Please provide your recognized key in the format of <<KEY>> at the first word of your response, followed by your detailed analysis. For example, if you recognized the provided frequency series as key "6", then the first word of your response should be <<6>>.
                    """)
        else:
            if enable_freq:
                return textwrap.dedent("""
                    Please provide your answer in the following JSON structure:
                    {
                        "frequencies": [list of peaks frequencies as floating-point numbers rounded to two decimal places],
                        "key": "the recognized key",
                        "analysis": "a detailed explanation of your analysis process."
                    }
                    """)
            else:
                return textwrap.dedent("""
                    Please provide your answer in the following JSON structure:
                    {
                        "key": "the recognized key",
                        "analysis": "a detailed explanation of your analysis process."
                    }
                    """)

    
    def get_dtmf_map(self):
        return textwrap.dedent("""
        | Low Frequency (Hz)  | High Frequency (Hz)| Key  |
        |---------------------|--------------------|------|
        | 697                 | 1209               | 1    |
        | 697                 | 1336               | 2    |
        | 697                 | 1477               | 3    |
        | 770                 | 1209               | 4    |
        | 770                 | 1336               | 5    |
        | 770                 | 1477               | 6    |
        | 852                 | 1209               | 7    |
        | 852                 | 1336               | 8    |
        | 852                 | 1477               | 9    |
        | 941                 | 1209               | *    |
        | 941                 | 1336               | 0    |
        | 941                 | 1477               | #    |
        """)

    def get_freq_text_prompt(self, freq_mag_text, model, enable_guide=True):
        """
        Generate prompt for input of frequency signals series text.
        Parameters:
            freq_mag_text: Frequency signals series in the format of 'frequency1:magnitude1, frequency2:magnitude2, ...'
            enable_guide: Whether add step-by-step guidance (i.e, tell the models first identify the most prominent frequency peaks before mapping them to their corresponding DTMF key) in prompt. If not enable_guide, the response format prompt will not include "frequencies" to avoid any relative hints. Otherwise, the models will be prompted to  return both key and frequencies. Default to `True`.
        """
        if enable_guide:
            return (
                self.system_prompt +
                "You are given a series of frequency components with corresponding amplitudes in the format of \"frequency1:magnitude1, frequency2:magnitude2, ...\". The frequency series represents a key (0-9, * or #). " +
                "Please first recognize the frequencies with high magnitudes in this series. Then, identify the key it refers to. " +
                f"Here is the frequency series:  {freq_mag_text}" +
                self.get_response_format_prompt(model=model, enable_freq=enable_guide)
            )
        else:
            return (
                self.system_prompt +
                f"You are given a series of frequency components with corresponding amplitudes in the format of \"frequency1:magnitude1, frequency2:magnitude2, ...\". The frequency series represents a key (0-9, * or #). Please identify the key it refers to. Here is the frequency series:  {freq_mag_text}" +
                self.get_response_format_prompt(model=model, enable_freq=enable_guide)
            )
    
    def get_freq_plot_prompt(self, enable_guide=True):
        """Generate prompt for input of frequency line plot"""
        return self.system_prompt + "Please first recognize the exact frequencies of peaks in this plot. Then, identify the key it refers to." + self.get_response_format_prompt(model="gpt-4o", enable_freq=enable_guide)
    
    def get_freq_pair_prompt(self, freq_pair, model, enable_map=False):
        """
        Generate prompt for input of a low- and high-frequency pair of the tone of a key.
        Parameters:
            freq_pair: frequency pair in the formate of '{low_frequency} Hz, {high_frequency} Hz'
            enable_map: whether to include the correct DTMF map in the prompt
        """
        if enable_map:
            return (
                self.system_prompt + 
                f"You are given a frequency pair of a tone. This tone represents a key (0-9, * or #). Here is the frequency pair: {freq_pair}. Please identify the key it refers to using the map below." + 
                self.get_dtmf_map() + 
                self.get_response_format_prompt(model=model, enable_freq=False)
            )
        else:
            return (
                self.system_prompt + 
                f"You are given a frequency pair of a tone which represents a key (0-9, * or #). Please identify the key it refers to. Here is the frequency pair: {freq_pair}" + 
                self.get_response_format_prompt(model=model, enable_freq=False)
            )



class PromptConfigHAR:
    """Configuration class storing device parameters and generating prompts for human activity recognition."""

    def __init__(self, location, freq):
        """Initialize fixed parameters to avoid redundant passing."""
        self.system_prompt = f"You are an expert of IMU-based human activity analysis. The IMU data is collected from smartphone attached to the user's {location} for 10 seconds with a sampling rate of {freq}Hz. "

    def get_response_format_prompt(self, model):
        """Generate prompt for different response format: json and raw text. For deepseek-reasoner (DeepSeek R1), since it doesn't support json output, prompt it to response in raw text with the classified label braced in <<>>; for all other models prompt them to return in json format with keys "label" and "analysis" """
        if model == "deepseek-reasoner":
            # deepseek-reasoner (DeepSeek R1) doesn't support json response
            return "Please provide your answer in the format of <<ACTION>> at the first word of your response, followed by your detailed analysis."
        else:
            return textwrap.dedent("""
            Please provide your answer in the following JSON structure:
            {
                "label": the predicted human action category,
                "analysis": a detailed explanation of your analysis process
            }
            """)

    def get_time_text_prompt(self, imu_data, model):
        """Generate prompt for IMU time series text input."""
        return (
            self.system_prompt +
            textwrap.dedent(f"""
                The person's action belongs to one of the following categories: Still, Walking, Run, Car. The IMU data is given in the IMU coordinate frame. The three-axis accelerations, gyroscopes and magnetometers are given below.
                Accelerations:
                x-axis: {imu_data.iloc[0]}; y-axis: {imu_data.iloc[1]}; z-axis: {imu_data.iloc[2]}
                Gyroscopes:
                x-axis: {imu_data.iloc[3]}; y-axis: {imu_data.iloc[4]}; z-axis: {imu_data.iloc[5]}
                Magnetometer:
                x-axis: {imu_data.iloc[6]}; y-axis: {imu_data.iloc[7]}; z-axis: {imu_data.iloc[8]}
                Could you please tell me what action the person was doing based on the given information and IMU readings? Please make an analysis step by step.
            """) +
            self.get_response_format_prompt(model)
        )

    def get_time_text_fewshot_prompt(self, fewshot_data_dict, imu_data, model):
        """Generate prompt for IMU time series text with one example for each class."""
        return (
            self.system_prompt +
            textwrap.dedent(f"""
            The person's action belongs to one of the following categories: Still, Walking, Run, Car. Here are examples for each class:
            Still:
                Accelerations:
                x-axis: {fewshot_data_dict['Still'].iloc[0]}; y-axis: {fewshot_data_dict['Still'].iloc[1]}; z-axis: {fewshot_data_dict['Still'].iloc[2]}
                Gyroscopes:
                x-axis: {fewshot_data_dict['Still'].iloc[3]}; y-axis: {fewshot_data_dict['Still'].iloc[4]}; z-axis: {fewshot_data_dict['Still'].iloc[5]}
                Magnetometer:
                x-axis: {fewshot_data_dict['Still'].iloc[6]}; y-axis: {fewshot_data_dict['Still'].iloc[7]}; z-axis: {fewshot_data_dict['Still'].iloc[8]}
            Walking:
                Accelerations:
                x-axis: {fewshot_data_dict['Walking'].iloc[0]}; y-axis: {fewshot_data_dict['Walking'].iloc[1]}; z-axis: {fewshot_data_dict['Walking'].iloc[2]}
                Gyroscopes:
                x-axis: {fewshot_data_dict['Walking'].iloc[3]}; y-axis: {fewshot_data_dict['Walking'].iloc[4]}; z-axis: {fewshot_data_dict['Walking'].iloc[5]}
                Magnetometer:
                x-axis: {fewshot_data_dict['Walking'].iloc[6]}; y-axis: {fewshot_data_dict['Walking'].iloc[7]}; z-axis: {fewshot_data_dict['Walking'].iloc[8]}
            Run:
                Accelerations:
                x-axis: {fewshot_data_dict['Run'].iloc[0]}; y-axis: {fewshot_data_dict['Run'].iloc[1]}; z-axis: {fewshot_data_dict['Run'].iloc[2]}
                Gyroscopes:
                x-axis: {fewshot_data_dict['Run'].iloc[3]}; y-axis: {fewshot_data_dict['Run'].iloc[4]}; z-axis: {fewshot_data_dict['Run'].iloc[5]}
                Magnetometer:
                x-axis: {fewshot_data_dict['Run'].iloc[6]}; y-axis: {fewshot_data_dict['Run'].iloc[7]}; z-axis: {fewshot_data_dict['Run'].iloc[8]}
            Car:
                Accelerations:
                x-axis: {fewshot_data_dict['Car'].iloc[0]}; y-axis: {fewshot_data_dict['Car'].iloc[1]}; z-axis: {fewshot_data_dict['Car'].iloc[2]}
                Gyroscopes:
                x-axis: {fewshot_data_dict['Car'].iloc[3]}; y-axis: {fewshot_data_dict['Car'].iloc[4]}; z-axis: {fewshot_data_dict['Car'].iloc[5]}
                Magnetometer:
                x-axis: {fewshot_data_dict['Car'].iloc[6]}; y-axis: {fewshot_data_dict['Car'].iloc[7]}; z-axis: {fewshot_data_dict['Car'].iloc[8]}
            The IMU data is given in the IMU coordinate frame. The three-axis accelerations, gyroscopes and magnetometers are given below.
                Accelerations:
                x-axis: {imu_data.iloc[0]}; y-axis: {imu_data.iloc[1]}; z-axis: {imu_data.iloc[2]}
                Gyroscopes:
                x-axis: {imu_data.iloc[3]}; y-axis: {imu_data.iloc[4]}; z-axis: {imu_data.iloc[5]}
                Magnetometer:
                x-axis: {imu_data.iloc[6]}; y-axis: {imu_data.iloc[7]}; z-axis: {imu_data.iloc[8]}
                Could you please tell me what action the person was doing based on the given information and IMU readings? Please make an analysis step by step.
            """) +
            self.get_response_format_prompt(model)
        )

    def get_time_text_description_step1_prompt(self, imu_data):
        """Generate prompt for IMU time series description."""
        return (
            self.system_prompt +
            textwrap.dedent(f"""
                The IMU data is given in the IMU coordinate frame. The three-axis accelerations, gyroscopes and magnetometers are given below.
                Accelerations:
                x-axis: {imu_data.iloc[0]}; y-axis: {imu_data.iloc[1]}; z-axis: {imu_data.iloc[2]}
                Gyroscopes:
                x-axis: {imu_data.iloc[3]}; y-axis: {imu_data.iloc[4]}; z-axis: {imu_data.iloc[5]}
                Magnetometer:
                x-axis: {imu_data.iloc[6]}; y-axis: {imu_data.iloc[7]}; z-axis: {imu_data.iloc[8]}
                Please analyse the sensor data trend variations, and provide a summary of sensor data's main elements and their trend distributions.
            """)
        )

    def get_time_text_description_step2_prompt(self, imu_desc, imu_data, model):
        """Generate prompt for classification with IMU time series text and description."""
        return (
            self.system_prompt +
            textwrap.dedent(f"""
                The person's action belongs to one of the following categories: Still, Walking, Run, Car. The IMU data is given in the IMU coordinate frame.
                Here is a detailed description of the IMU signals: {imu_desc}.
                The raw three-axis accelerations, gyroscopes and magnetometers are given below for your reference.
                    Accelerations:
                    x-axis: {imu_data.iloc[0]}; y-axis: {imu_data.iloc[1]}; z-axis: {imu_data.iloc[2]}
                    Gyroscopes:
                    x-axis: {imu_data.iloc[3]}; y-axis: {imu_data.iloc[4]}; z-axis: {imu_data.iloc[5]}
                    Magnetometer:
                    x-axis: {imu_data.iloc[6]}; y-axis: {imu_data.iloc[7]}; z-axis: {imu_data.iloc[8]}
                    Could you please tell me what action the person was doing based on the given information, IMU description and IMU readings? Please make an analysis step by step.
            """) +
            self.get_response_format_prompt(model)
        )

    def get_time_plot_prompt(self):
        """Generate prompt for IMU time series plot."""
        return (
            self.system_prompt +
            textwrap.dedent(f"""
                The three-axis accelerations, gyroscopes and magnetometers are plotted in the given plot.
                The person's action belongs to one of the following categories: Still, Walking, Run, Car.
                Could you please tell me what action the person was doing? Please make an analysis step by step.
            """) +
            self.get_response_format_prompt(model="gpt-4o")
        )

    def get_time_plot_fewshot_prompt(self):
        """Generate prompt for IMU time series plot with one example for each class."""
        return (
            self.system_prompt +
            textwrap.dedent(f"""
                The three-axis accelerations, gyroscopes and magnetometers are plotted in the given plot.
                The person's action belongs to one of the following categories: Still, Walking, Run, Car. The first four plots are examples of IMU data readings for Still, Walking, Run and Car respectively.
                Could you please tell me what action the person was doing based on the last plot? Please make an analysis step by step.
            """) +
            self.get_response_format_prompt(model="gpt-4o")
        )

    def get_time_plot_env_prompt(self):
        """Generate prompt for IMU time series plot with environmental photo taken during the user's transportation."""
        return (
            self.system_prompt + 
            textwrap.dedent(f"""
                You are given 2 images: the first image is the plot of three-axis accelerations, gyroscopes and magnetometers; the second image is an envorinment photo taken during this activity by a camera placed in the user's torso.
                The person's action belongs to one of the following categories: Still, Walking, Run, Car.
                Could you please tell me what action the person was doing based on the given information, IMU readings and environment photo? Please make an analysis step by step.
            """) +
            self.get_response_format_prompt(model="gpt-4o")
        )
    
    def get_env_only_prompt(self):
        """Generate prompt for IMU time series plot with environmental photo taken during the user's transportation."""
        return (
            textwrap.dedent(f"""
                You are an expert in human activity analysis. You are given an envorinment photo taken during a person's transportation by a camera placed in their torso.
                The person's transportation method belongs to one of the following categories: Still, Walking, Run, Car.
                Could you please tell me what transportation method the person was using based on the given information and the environment photo? Please make an analysis step by step.
            """) +
            self.get_response_format_prompt(model="gpt-4o")
        )
