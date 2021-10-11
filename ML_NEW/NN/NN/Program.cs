
using System;
using System.Linq;
using NN;

static class MainClass
{
    private const String SAVE_FOLDER_NAME = "SAVED_NETWORKS";

    static void Main(string[] args)
    {
        // run the main state if initialization succeeds 
        if ( Init() )
            State_Main();
    }

    /// <summary>
    /// Ensures the environment is okay for execution
    /// </summary>
    static bool Init ()
    {
        bool success = true;

        Console.WriteLine("Initialization Log:");

        // make save folder directory
        String curDir = System.IO.Directory.GetCurrentDirectory();
        String saveFilePath = System.IO.Path.Combine(curDir, SAVE_FOLDER_NAME);

        if (!System.IO.Directory.Exists(saveFilePath))
        { 
            FileManager.MakeDirectory(SAVE_FOLDER_NAME, curDir);
            Console.WriteLine("No save folder found... Making save folder");
        }
        else
        {
            Console.WriteLine("Save folder found");
        }
    
        // ensure MNIST dataset is available
        if (MNISTReader.CanAccessMNISTData ())
        {
            Console.WriteLine("MNIST dataset is accessible");
        }
        else
        {
            UI.DisplayCentered("UNABLE TO START PROGRAM!");
            UI.Display_Error("Unable to find MNIST dataset....\nMake sure the dataset folder ('MNIST_Dataset') is located in the same directory as the program executable", true);
            success = false;
        }

        return success;
    }

    /// <summary>
    /// Main state of the program, handles user interaction
    /// </summary>
    static void State_Main ()
    {
        bool flag_exit = false;
        bool flag_loaded_network = false;

        NeuralNetwork network = null;
        TrainingParams tParams = new TrainingParams ();

        String [] options = {
            "Load Network",
            "Create new Network",
            "Train network",
            "Test Network",
            "Save Network",
            "Quit"
        };
        
        while (!flag_exit)
        {
            // network status
            String currentNetworkDescription = "No network loaded! load a network to continue";
            if (flag_loaded_network)
            {
                currentNetworkDescription = "Network format: " + network.Inputs.Rows.ToString ();

                for (int i = 0; i < network.Layers.Length; i ++ )
                {
                    currentNetworkDescription += " ";
                    currentNetworkDescription += network.Layers [i].Size;
                }
            }

            // display options
            int response = UI.MenuTree ("\tMAIN MENU\t", options, currentNetworkDescription);
            
            // handle user input
            switch (response)
            {
                // LOAD
                case 1:

                    if (flag_loaded_network)
                    {
                        // warn user they will loose the current network they are working on
                        Console.WriteLine ("Do you want to load a new Neural network?");
                        Console.WriteLine ("This will replace the network you are currently using");
                        bool answer = UI.YesNoQuestion (true, true);

                        if (answer)
                            network = State_Load ();
                    }
                    else
                        network = State_Load ();

                    // set network loaded flag
                    if (network != null)
                        flag_loaded_network = true;
                    else
                        flag_loaded_network = false;

                    break;

                // CREATE
                case 2:

                    if (flag_loaded_network)
                    {
                        // warn user they will loose the current network they are working on
                        Console.WriteLine ("Do you want to make a new Neural network?");
                        Console.WriteLine ("This will replace the network you are currently using");
                        bool answer = UI.YesNoQuestion (true, true);

                        if (answer)
                            network = State_Create ();
                    }
                    else
                        network = State_Create ();

                    // set network loaded flag
                    if (network != null)
                        flag_loaded_network = true;
                    else
                        flag_loaded_network = false;
                break;

                // TRAIN
                case 3:
                    if (flag_loaded_network)
                        State_Train (network, tParams);
                    else
                        UI.Display_Error ("Must load a network to train");
                break;
            
                // TEST
                case 4:
                    if (flag_loaded_network)
                        State_Test (network);
                    else
                        UI.Display_Error ("Must load a network to test");
                break;

                // SAVE
                case 5:
                    if (flag_loaded_network)
                        State_Save (network);
                    else
                        UI.Display_Error ("Load a network before saving");
                break;

                // EXIT
                case 6:
                    flag_exit = true;
                break;
            }

        }
    }

    // handle user interaction to load a saved network
    static NeuralNetwork State_Load ()
    {
        Console.WriteLine ("Enter the filename of the saved Neural Network: (.json)");
        UI.AssertContinue ();
        String fileName = Console.ReadLine ();

        if (!fileName.Contains (".json"))
            fileName = fileName + ".json";

        Console.WriteLine ("Loading network:" + fileName);

        String filePath = SAVE_FOLDER_NAME + "/" + fileName;

        NeuralNetwork net = FileManager.DeserializeJSON (filePath);

        return net;
    }

    // handle user interaction to create a network
    static NeuralNetwork State_Create ()
    {
        NeuralNetwork net;
        Console.WriteLine("Enter the hidden layer sizes");
        Console.WriteLine ("E.G: '30 20' = 2 hidden layers with 30 and 20 neurons respectively");
        Console.WriteLine ("'50 20 15' = 3 hidden layers with 50 20 and 15 neurons respectively");

        // read user input
        String netFormat = Console.ReadLine ();

        // ensure format string is correct
        if ( !netFormat.Equals ("") )
            netFormat = "784 " + netFormat + " 10";
        else
            netFormat = "784 10";

        

        Console.WriteLine ("Randomise weights");
        bool randomise = UI.YesNoQuestion (true);
        
        net = new NeuralNetwork (netFormat, randomise);

        if (net.isEmpty())
        {
            net = null;
            UI.Display_Error("Incorrect format: " + netFormat);
        }
        else
            Console.WriteLine("Network structure: " + netFormat);
        
        return net;
    }

    // handle user interaction to save a network
    static void State_Save (NeuralNetwork network)
    {
        Console.WriteLine ("Enter the filename to save the Neural Network as: (.json)");
        UI.AssertContinue ();

        String fileName = Console.ReadLine ();

        if (!fileName.Contains (".json"))
            fileName = fileName + ".json";

        // save in the network saves folder
        String filePath = SAVE_FOLDER_NAME + "/" + fileName; 

        Console.WriteLine ("Saving as: " + fileName);
        FileManager.SerializeAsJson (network, filePath);
        Console.WriteLine ("Saved!");
        UI.AssertContinue ();
    }

    // menu state with options for training neural networks
    static void State_Train (NeuralNetwork network, TrainingParams tParams)
    {
        bool flag_exit = false;

        String[] menu_options = {
            "Train",
            "Train for one epoch",
            "Change Training HyperParameters",
            "Exit To Main Menu"
        };

        // menu interaction loop 
        while (!flag_exit)
        {
            Console.Clear ();

            // display menu 
            int response = UI.MenuTree ("Training Menu", menu_options, "Hyperparameters: " + tParams.ToString () + "\n");

            // handle response
            switch (response)
            {
                // Train
                case 1:
                    Console.WriteLine ("Training Network: ");
                    TrainNetwork (network, tParams);
                    Console.WriteLine ("Training Done!");
                    UI.AssertContinue ();
                break;

                // Train for one Epoch
                case 2:
                    // copy training parameters and adjust them to only have one epoch 
                    TrainingParams oneEpochParams = new TrainingParams (tParams);
                    oneEpochParams.MaxEpoch = 1;

                    TrainNetwork (network, oneEpochParams);
                    Console.WriteLine ("Training Done!");
                    UI.AssertContinue ();
                break;

                // Change Training Parameters
                case 3:
                    State_TrainingParams_Menu(tParams);
                break;

                // Exit
                case 4:
                    flag_exit = true;
                break;
            }
        }
    }

    // Handle user interaction with changing TrainingParams for network training
    static TrainingParams State_TrainingParams_Menu(TrainingParams tParams)
    {
        bool flag_exit = false;

        String[] menu_options =
        {
            "Learning Rate",
            "Momentum",
            "Target Cost",
            "Max Epoch",
            "Batch Size",
            "Revert to Default HyperParameters",
            "Exit To Training Menu"
        };

        // menu interaction loop
        while (!flag_exit)
        {
            // display menu
            int response = UI.MenuTree("Select Training HyperParameter to change: ", menu_options, tParams.ToString());

            // handle response
            switch (response)
            {
                // learning rate
                case 1:
                    float newLearningRate = UI.Query_Float("Enter the new learning rate:", 0.0001f, 1);
                    break;

                // momentum
                case 2:
                    float newMomentum = UI.Query_Float("Enter the new momentum value:", 0, 1);
                    tParams.Momentum = newMomentum;
                    break;

                // target cost
                case 3:
                    float newTargetCost = UI.Query_Float("Enter the new target cost:", 0.01f, 1);
                    tParams.TargetCost = newTargetCost;
                    break;

                // max epoch
                case 4:
                    int newMaxEpoch = UI.Query_Int("Enter new max Epoch:", 0, 10);
                    tParams.MaxEpoch = newMaxEpoch;
                    break;

                // batch size
                case 5:
                    int newBatchSize = UI.Query_Int("Enter new batch size:", 1, 10000);
                    tParams.BatchSize = newBatchSize;
                    break;

                // restore defaults
                case 6:
                    tParams = new TrainingParams();
                    break;

                // exit
                case 7:
                    flag_exit = true;
                    break;
            }
        }

        return tParams;
    }

    // Handle user interaction for testing a network
    static void State_Test (NeuralNetwork network)
    {
        Console.WriteLine ("Commencing network test: ");
        UI.SkpLn (2);
        float testAccuracy = MNISTOptimiser.TestNetwork (network);
        Console.Clear();
        Console.WriteLine ("Testing done!");
        Console.WriteLine("Accuracy: {0}%", testAccuracy);
        UI.AssertContinue ();
    }

    // Handle user interaction for training a network
    static void TrainNetwork (NeuralNetwork net, TrainingParams trainingParams)
    {
        // ask if should log training 
        Console.Clear();
        Console.WriteLine("Log training steps to csv file? (y/n)");

        bool answer = UI.YesNoQuestion();
        String logFileName = null;

        if (answer)
        {
            Console.WriteLine("Gradient descent steps will be logged");
            Console.WriteLine("Enter the file name (.csv): ");
            logFileName = Console.ReadLine();

            // ensure output file is of type .csv
            if (!logFileName.Contains (".csv"))
            {
                logFileName.Concat (".csv");
            }
        }

        Console.Clear();

        // train network
        MNISTOptimiser.TrainNetwork(ref net, trainingParams, logFileName);
    }
}