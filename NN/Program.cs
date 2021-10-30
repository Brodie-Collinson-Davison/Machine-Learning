
using System;
using System.Linq;
using System.IO;
using NN;

static class MainClass
{
    private const string SAVE_FOLDER_NAME = "Saved_Networks";

    static void Main(string[] args)
    {
        // ensures environment is correct before executing program
        //Initialize();
        // start the program
        //RunProgramLoop();

        NeuralNetwork net = new NeuralNetwork("784 32r 10r");

        TrainingParams tParams = new TrainingParams();
        tParams.BatchSize = 50;
        tParams.MaxEpoch = 1;
        tParams.Momentum = 0.25f;
        tParams.LearningRate = 0.01f;

        MNISTOptimiser.TrainNetwork(net, tParams);
        Console.WriteLine(MNISTOptimiser.TestNetwork(net));
        Console.ReadLine();
    }

    private static void RunProgramLoop()
    {
        bool exitFlag = false;

        while (!exitFlag)
        {
            string[] options = {
                "Load Network",
                "Save Network",
                "Test",
                "Train",
                "Exit"
            };

            // display menu options and get user 
            int menuSelection = UI.MenuTree("Main Menu", options);

            switch (menuSelection)
            {
                // Load Network
                case 1:
                    NeuralNetwork net = LoadNetwork();
                    break;

                // Save Network
                case 2:

                    break;

                // Test
                case 3:

                    break;

                // Train
                case 4:

                    break;

                // Exit
                case 5:
                    exitFlag = true;
                    break;
            }
        }
    }

    private static void Initialize ()
    { 
        if (!SearchForSaveFolder())
        {
            // create a folder for saving networks in the above directory
            string parentPath = Directory.GetParent(Directory.GetCurrentDirectory()).FullName;
            Directory.CreateDirectory(Path.Combine(parentPath, SAVE_FOLDER_NAME));
        }
    }

    private static NeuralNetwork LoadNetwork()
    {
        NeuralNetwork net = null;

        string[] savedFileNames = FindSaveFileNames();        

        // display names
        UI.ListSelection("Select the network to load", savedFileNames);
        // load selected network

        return net;
    }

    private static string [] FindSaveFileNames ()
    {
        // find saved neural network names
        string saveFilePath = Path.Combine(Directory.GetParent(Directory.GetCurrentDirectory()).FullName, SAVE_FOLDER_NAME);
        string[] fileNames = Directory.GetFiles(saveFilePath);

        for (int i = 0; i < fileNames.Length; i++)
        {
            string str = fileNames[i];
            str = str.Substring(str.LastIndexOf('\\') + 1);
            fileNames[i] = str;
        }

        return fileNames;
    }

    private static bool SearchForSaveFolder ()
    {
        bool result = false;

        string current = Directory.GetCurrentDirectory();

        // serach directory above
        if (Directory.Exists (Path.Combine (Directory.GetParent (current).FullName, SAVE_FOLDER_NAME)))
            result = true;

        return result;
    }
}