import os
import sys
import argparse
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import random
import csv
import pandas as pd


class DtmfGenerator:
    DTMF_TABLE = {
        "1": np.array([1209, 697]),
        "2": np.array([1336, 697]),
        "3": np.array([1477, 697]),
        "A": np.array([1633, 697]),
        "4": np.array([1209, 770]),
        "5": np.array([1336, 770]),
        "6": np.array([1477, 770]),
        "B": np.array([1633, 770]),
        "7": np.array([1209, 852]),
        "8": np.array([1336, 852]),
        "9": np.array([1477, 852]),
        "C": np.array([1633, 852]),
        "*": np.array([1209, 941]),
        "0": np.array([1336, 941]),
        "#": np.array([1477, 941]),
        "D": np.array([1633, 941]),
    }

    def __init__(
        self,
        phone_number: str,
        Fs: float,
        time: float,
        delay: float,
        amp: float,
        noise_level: float,
        freq_file: str,
        folder_path: str
    ):
        self.freq_file = freq_file
        self.folder_path = folder_path
        self.signal = self.compose(phone_number, Fs, time, delay, amp, noise_level)

    def __dtmf_function(
        self, number: str, Fs: float, time: float, delay: float, amp: float, noise_level: float
    ) -> np.array:
        """
        Function which generate DTMF tone (samples) to one specific character
        and its delay

        :number: Represents the character to be converted to DTMF tone
        :Fs: Sample frequency used to generate the signal in Hz
        :time: Duration of each tone in seconds
        :delay: Duration of delay between each tone in seconds
        :amp: Amplitude of the sine waves
        :noise_level: Noise level to add to the signal (float between 0 and 1)

        :return: Array with samples to the DTMF tone and delay
        """

        time_tone = np.arange(0, time, (1 / Fs))
        time_delay = np.arange(0, delay, (1 / Fs))

        tone_samples = amp * (
            np.sin(2 * np.pi * self.DTMF_TABLE[number][0] * time_tone)
            + np.sin(2 * np.pi * self.DTMF_TABLE[number][1] * time_tone)
        )
        # Add Gaussian noise to the signal
        noise = noise_level * np.random.normal(0, 1, len(tone_samples))
        tone_samples_noise = tone_samples + noise
        # print(f"shape of tone_samples_noise: {tone_samples_noise.shape}, {tone_samples_noise}")
        # input("Press Enter...")

        primary_freqs = self.extract_frequencies(tone_samples_noise, Fs)
        '''
        file_exists = os.path.isfile(self.freq_file)
        with open(self.freq_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["number", "Freq1", "Freq2"])
            writer.writerow([number, primary_freqs[0], primary_freqs[1]])
        '''
        # input("Press Enter...")
        delay_samples = np.sin(2 * np.pi * 0 * time_delay)
        # print(delay_samples.shape, delay_samples)
        # input("Press Enter...")

        return np.append(tone_samples, delay_samples)

    def extract_frequencies(self, signal, Fs):
        """
        Extract two main frequencies from noised tone sample
        """
        fft_spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1 / Fs)
        magnitude = np.abs(fft_spectrum)

        # Keep positive frequencies only
        positive_freqs = freqs > 0
        freqs = freqs[positive_freqs]
        magnitude = magnitude[positive_freqs]

        # Keep the 2 largest frequencies
        peak_indices = magnitude.argsort()[-2:][::-1]
        return sorted([freqs[peak_indices[0]], freqs[peak_indices[1]]])


    def compose(
        self,
        phone_number: str,
        Fs: float,
        time: float,
        delay: float,
        amp: float,
        noise_level: float,
    ) -> np.array:
        """
        Function which generate DTMF tones (samples) to compose a signal
        representing the phone number

        :number: Represents the number to be converted to DTMF tone
        :Fs: Sample frequency used to generate the signal in Hz
        :time: Duration of each tone in seconds
        :delay: Duration of delay between each tone in seconds
        :amp: Amplitude of the sine waves
        :noise_level: Noise level to add to the signal (float between 0 and 1)

        :return: Array with samples to the DTMF tone and delay
        """

        signal = np.array([])

        for number in phone_number:
            tone_delay_signal = self.__dtmf_function(number, Fs, time, delay, amp, noise_level)
            signal = np.append(signal, tone_delay_signal)

        return signal

    def test_signal(
        self,
        filename: str,
        phone_number: str,
        Fs: float,
        time: float,
        delay: float,
        add_grid_lines: bool = False,
    ):
        """
        Function which debug DTMF tones generated in the WAV file plotting their frequencies

        :filename: WAV filename to debug
        :phone_number: Phone number to verify
        :Fs: Sample frequency used to generate the signal in Hz
        :time: Duration of each tone in seconds
        :delay: Duration of delay between each tone in seconds

        :return: A graph with tones and their frequencies
        """
        # Read the WAV file
        rate, signal = wav.read(filename)

        # Validate sampling rate
        if rate != Fs:
            raise ValueError(f"Sampling rate in WAV file ({rate} Hz) does not match expected rate ({Fs} Hz).")

        # Compute the expected segment length for each tone (including delay)
        segment_length = int(Fs * (time + delay))
        n_tones = len(phone_number)

        # Initialize the figure layout
        n_columns = int(np.ceil(np.sqrt(n_tones)))
        n_rows = int(np.ceil(n_tones / n_columns))
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(12, 8))
        axes = np.array(axes).flatten()

        # save frequencies text results
        all_freqs = []

        # Process each tone
        for i in range(n_tones):
            start_idx = i * segment_length
            end_idx = start_idx + int(Fs * time)  # Only consider the tone, exclude delay

            # Handle cases where the signal may be shorter than expected
            if start_idx >= len(signal):
                axes[i].set_title(f"Tone {i + 1}")
                axes[i].axis("off")
                continue

            if end_idx > len(signal):
                end_idx = len(signal)  # Adjust if the last tone is incomplete

            # Extract the segment for the tone
            segment = signal[start_idx:end_idx]

            # Compute FFT
            tone_fft = np.fft.fft(segment)
            freq = np.fft.fftfreq(len(tone_fft), d=1 / Fs)
            magnitude = np.abs(tone_fft)
            
            # Filter only positive frequencies
            positive_freqs = freq > 0
            freq = freq[positive_freqs]
            magnitude = magnitude[positive_freqs]

            for f, mag in zip(freq, magnitude):
                all_freqs.append([i, f, mag])  # i is the No. of windows
            
            # Plot the spectrum
            axes[i].plot(freq, magnitude)
            axes[i].set_xlim(0, 2000)  # DTMF frequency range
            axes[i].set_title(f"Tone {i + 1}")
            axes[i].set_xlabel("Frequency (Hz)")
            axes[i].set_ylabel("Amplitude")

            # Save each tone in separate plots
            plt.figure()
            plt.plot(freq, magnitude)
            plt.xlim(0, 2000)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")

            # # Save raw individual tone plot
            plt.savefig(f"{self.folder_path}/{phone_number}_tone{i+1}_freqs.png")
            
            if add_grid_lines:
                plt.minorticks_on()
                plt.grid(True, which='both', linewidth=0.5)
                plt.savefig(f"{self.folder_path}/{phone_number}_tone{i+1}_freqs_grid.png")

            # # Save zoomed-in individual tone plot (600-2000 Hz)
            # plt.xlim(600, 1600)
            # plt.grid(False, which='both')  # Reset grid
            # plt.savefig(f"{self.folder_path}/{phone_number}_tone{i+1}_freqs_zoomed.png")

            # if add_grid_lines:
            #     plt.minorticks_on()
            #     plt.grid(True, which='both', linewidth=0.5)
            #     plt.savefig(f"{self.folder_path}/{phone_number}_tone{i+1}_freqs_zoomed_grid.png")

            plt.close()

        # Turn off unused subplots
        for j in range(n_tones, len(axes)):
            axes[j].axis("off")
        
        
        # Set the overall title and save the plot
        plt.tight_layout()
        # plt.savefig(f"{self.folder_path}/{phone_number}_freqs.png")

        if add_grid_lines:
            for ax in axes:
                # ax.set_xticks(np.arange(0, 2000, 250))
                ax.minorticks_on()
                ax.grid(True, which='both', linewidth=0.5)
            plt.savefig(f"{self.folder_path}/{phone_number}_freqs_grid.png")

        # # Save plots with frequencies only in 600-1600
        # for ax in axes:
        #     ax.set_xlim(600, 1600)
        #     ax.grid(False, which='both') # reset grid lines

        # plt.savefig(f"{self.folder_path}/{phone_number}_freqs_zoomed.png")

        if add_grid_lines:
            for ax in axes:
                ax.minorticks_on()
                ax.grid(True, which='both', linewidth=0.5)  # Ensure grid for zoomed plots
            plt.savefig(f"{self.folder_path}/{phone_number}_freqs_zoomed_grid.png")  # Zoomed range with grid
        plt.close()
        
        # Save frequency text results to CSV
        df = pd.DataFrame(all_freqs, columns=["Tone", "Frequency (Hz)", "Magnitude"])
        df.to_csv(f"{self.folder_path}/{phone_number}_freqs.csv", index=False)


def main():
    try:
        parser = argparse.ArgumentParser(description="DTMF generator to phone numbers.")
        parser.add_argument(
            "-f",
            "--samplefrequency",
            required=True,
            type=int,
            help="Sample Frequency (Hz)",
        )
        parser.add_argument(
            "-t",
            "--toneduration",
            required=True,
            type=float,
            help="Tones duration (s)",
        )
        parser.add_argument(
            "-s",
            "--silence",
            required=True,
            type=float,
            help="Silence duration between tones duration (s)",
        )
        parser.add_argument(
            "-a",
            "--amplitude",
            required=True,
            type=float,
            help="Amplitude of the sine waves",
        )
        parser.add_argument(
            "-nl",
            "--noiselevel",
            type=float,
            default=0.05,
            help="Noise level to add to the signal (float between 0 and 1)",
        )
        parser.add_argument(
            "-d",
            "--debug",
            action="store_true",
            help="Enable FFT graph of each tone (character) to debug",
        )
        parser.add_argument(
            "-n",
            "--number",
            type=int,
            help="Number of random phone numbers to generate",
        )
        args = parser.parse_args()

        choices = [str(i) for i in range(10)] + ["*", "#"]
        # for _ in range(args.number):
        #     phonenumber = ''.join([random.choice(choices) for _ in range(6)])

        folder_path = "/home/yihan/code/datasets/dtmf/dtmf_clean_12"

        ''' Temp '''
        # Define the folder path

        # Create an empty set to store unique filenames (without extension)
        wav_filenames = set()

        # Traverse the folder and collect .wav filenames
        for file in os.listdir("/home/yihan/code/datasets/dtmf/dtmf_noise_12"):
            if file.endswith(".wav"):
                filename_without_ext = os.path.splitext(file)[0]  # Remove .wav extension
                wav_filenames.add(filename_without_ext)
        ''''''

        for phonenumber in wav_filenames:
            # Generate dtmf frequency signals
            if os.path.isfile(f'{folder_path}/{phonenumber}_freqs.csv'):
                print(f"{phonenumber} already generated. Skip...")
                continue
            
            freq_file = f'{folder_path}/{phonenumber}_freqs_clean.csv'
            dtmf = DtmfGenerator(
                phonenumber,
                args.samplefrequency,
                args.toneduration,
                args.silence,
                args.amplitude,
                args.noiselevel,
                freq_file,
                folder_path
            )
            wav_output = f'{folder_path}/{phonenumber}.wav'
            print(f'len(dtmf.signal): {len(dtmf.signal)}')
            wav.write(wav_output, args.samplefrequency, dtmf.signal)

            # Plot generated signals to check, and save the plots (clean frequency domain signals plots )
            if args.debug:
                dtmf.test_signal(
                    wav_output,
                    phonenumber,
                    args.samplefrequency,
                    args.toneduration,
                    args.silence,
                    True
                )

            print(f"{phonenumber} done.")
            # input()
            
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
