package com.example.activitytracker.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.activitytracker.ui.screens.HistoryScreen
import com.example.activitytracker.ui.screens.MainScreen
import com.example.activitytracker.ui.screens.SettingsScreen
import com.example.activitytracker.ui.screens.SplashScreen
import com.example.activitytracker.viewmodel.MainViewModel

enum class Routes { Splash, Main, History, Settings }

@Composable
fun AppNavHost(vm: MainViewModel, navController: NavHostController = rememberNavController()) {
    NavHost(navController = navController, startDestination = Routes.Splash.name) {
        composable(Routes.Splash.name) {
            SplashScreen(onStart = { navController.navigate(Routes.Main.name) { popUpTo(Routes.Splash.name) { inclusive = true } } })
        }
        composable(Routes.Main.name) {
            MainScreen(
                vm = vm,
                onHistory = { navController.navigate(Routes.History.name) },
                onSettings = { navController.navigate(Routes.Settings.name) }
            )
        }
        composable(Routes.History.name) {
            HistoryScreen(onBack = { navController.popBackStack() }, vm = vm)
        }
        composable(Routes.Settings.name) {
            SettingsScreen(onBack = { navController.popBackStack() }, vm = vm)
        }
    }
}
