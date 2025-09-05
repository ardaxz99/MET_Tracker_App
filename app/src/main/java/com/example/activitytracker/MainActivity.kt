package com.example.activitytracker

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.room.Room
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import kotlinx.coroutines.launch
import com.example.activitytracker.data.db.AppDatabase
import com.example.activitytracker.data.repo.ActivityRepository
import com.example.activitytracker.navigation.AppNavHost
import com.example.activitytracker.prefs.SettingsStore
import com.example.activitytracker.sensors.SensorService
import com.example.activitytracker.ui.theme.ActivityTrackerTheme
import com.example.activitytracker.viewmodel.MainViewModel
import com.example.activitytracker.ml.CnnClassifier

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val db = Room.databaseBuilder(
            applicationContext,
            AppDatabase::class.java,
            "activity_tracker.db"
        ).fallbackToDestructiveMigration().build()

        val repo = ActivityRepository(db.sessionDao(), db.sliceDao(), db.secondDao())
        val settings = SettingsStore(applicationContext)
        val sensorService = SensorService(applicationContext)
        val classifier = try { CnnClassifier(applicationContext) } catch (t: Throwable) { null }

        val factory = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                @Suppress("UNCHECKED_CAST")
                return MainViewModel(repo, sensorService, settings, classifier) as T
            }
        }

        setContent {
            val vm: MainViewModel = viewModel(factory = factory)
            val dark by vm.themeDark.collectAsState()
            ActivityTrackerTheme(darkTheme = dark) {
                AppNavHost(vm)
            }
        }

        // Ensure window/stride defaults are set to 5s/1s for this build
        lifecycleScope.launch {
            settings.setWindowSeconds(5.0)
            settings.setStrideSeconds(1.0)
        }
    }
}
