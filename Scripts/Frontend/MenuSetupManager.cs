using System;

static class MenuCreator
{
    private static ConsoleRenderer renderer = new ConsoleRenderer();
    private static MenuManager manager  = new MenuManager(renderer);
    private static ButtonFactory factory  = new ButtonFactory(manager);

    static void DoMenuSetup()
    {
        manager.AddMenu(mainMenu);

        manager.ShowMenu(0);
    }

    private static void CreateStartMenu(){
        MenuScreen StartMenu = new MenuScreen(0);
    }

    private static void CreateLoadModelMenu(){
        MenuScreen ModelCreationMeny = new MenuScreen(1);
    }

    private static void CreateNewModelCreationMenu(){
        MenuScreen ModelCreationMeny = new MenuScreen(2);
    }

    private static void CreateInstructions(){
        MenuScreen InstructionsMenu = new MenuScreen(3);
    }

    private static void CreateLoadedModelMenu(){
        MenuScreen LoadedModelMenu = new MenuScreen(4);
    }

    private static void CreateModelParametersMenu(){
        MenuScreen ModelParametersMenu = new MenuScreen(5);
    }

    private static void CreateModelUsageMenu(){
        MenuScreen ModelUsageMenu = new MenuScreen(6);
    }

    private static void CreateTrainingFromScratchMenu(){
        MenuScreen TrainingFromScratchMenu = new MenuScreen(7);
    }

    private static void CreateAdditionalTrainingMenu(){
        MenuScreen AdditionalTrainingMenu = new MenuScreen(8);
    }

    private static void CreateTrainingLoopMenu(){
        MenuScreen TrainingLoopMenu = new MenuScreen(9);
    }
    
    private static void CreateModelInterferenceMenu(){
        MenuScreen ModelInterferenceMenu = new MenuScreen(10);
    }




    // private static void MakeMainMenuScreen(){
    //     MenuScreen mainMenu = new MenuScreen(0);
    //     mainMenu.AddButton(factory.Create(ButtonType.Redirect, "Instructions", null, 1));
    //     mainMenu.AddButton(factory.Create(ButtonType.Exit, "Exit"));
    // }

    // private static void MakeInstructionsScreen(){
    //     MenuScreen instructionsScreen = new MenuScreen(1);
    // }

    // private static void MakeModelCreationScreen(){
    //     MenuScreen modelCreationScreen = new MenuScreen(2);
    // }

    // private static void MakeTrainingScreen(){

    // }



}
