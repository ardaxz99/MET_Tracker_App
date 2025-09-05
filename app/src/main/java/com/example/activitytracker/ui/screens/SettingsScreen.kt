@file:OptIn(ExperimentalMaterial3Api::class)

package com.example.activitytracker.ui.screens

import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.ui.Alignment
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.RadioButton
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.activitytracker.viewmodel.MainViewModel
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.clickable

@Composable
fun SettingsScreen(onBack: () -> Unit, vm: MainViewModel) {
    val imu by vm.imuEnabled.collectAsState()
    val dark by vm.themeDark.collectAsState()
    val age by vm.ageYears.collectAsState()
    val weight by vm.weightKg.collectAsState()
    val height by vm.heightCm.collectAsState()
    val gender by vm.gender.collectAsState()

    Scaffold(topBar = {
        TopAppBar(title = { Text("Settings") }, navigationIcon = { IconButton(onClick = onBack) { Text("Back") } })
    }) { padding ->
        val scroll = rememberScrollState()
        Column(
            Modifier
                .padding(padding)
                .padding(16.dp)
                .fillMaxSize()
                .verticalScroll(scroll),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Dark mode")
                Switch(checked = dark, onCheckedChange = { vm.setThemeDark(it) })
            }
            // User Profile
            Text("User Profile")
            var editing by remember { mutableStateOf(false) }
            var ageText by remember(age) { mutableStateOf(if (age > 0) age.toString() else "") }
            var weightText by remember(weight) { mutableStateOf(if (weight > 0.0) String.format("%.1f", weight) else "") }
            var heightText by remember(height) { mutableStateOf(if (height > 0.0) String.format("%.1f", height) else "") }
            val genderOptions = listOf("Female", "Male")
            var genderIndex by remember(gender) {
                val idx = genderOptions.indexOfFirst { it.equals(gender, ignoreCase = true) }
                mutableStateOf(if (idx >= 0) idx else 1) // fallback to Male
            }

            OutlinedTextField(
                value = ageText,
                onValueChange = { if (it.length <= 3 && it.all { ch -> ch.isDigit() }) ageText = it },
                label = { Text("Age (years)") },
                singleLine = true,
                enabled = editing
            )
            OutlinedTextField(
                value = weightText,
                onValueChange = { new -> if (new.count { it == '.' } <= 1 && new.all { it.isDigit() || it == '.' }) weightText = new },
                label = { Text("Weight (kg)") },
                singleLine = true,
                enabled = editing
            )
            OutlinedTextField(
                value = heightText,
                onValueChange = { new -> if (new.count { it == '.' } <= 1 && new.all { it.isDigit() || it == '.' }) heightText = new },
                label = { Text("Height (cm)") },
                singleLine = true,
                enabled = editing
            )
            // Gender: show read-only field when not editing; when editing, show clear radio options.
            if (!editing) {
                OutlinedTextField(
                    readOnly = true,
                    value = genderOptions[genderIndex],
                    onValueChange = {},
                    label = { Text("Gender") },
                    singleLine = true,
                    enabled = false
                )
            } else {
                Text("Gender")
                Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    Row(Modifier.clickable { genderIndex = 0 }) {
                        RadioButton(selected = genderIndex == 0, onClick = { genderIndex = 0 })
                        Text("Female")
                    }
                    Row(Modifier.clickable { genderIndex = 1 }) {
                        RadioButton(selected = genderIndex == 1, onClick = { genderIndex = 1 })
                        Text("Male")
                    }
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                if (!editing) {
                    Button(onClick = { editing = true }) { Text("Edit") }
                } else {
                    Button(onClick = {
                        val ageVal = ageText.toIntOrNull() ?: 0
                        val weightVal = weightText.toDoubleOrNull() ?: 0.0
                        val heightVal = heightText.toDoubleOrNull() ?: 0.0
                        val genderVal = genderOptions[genderIndex]
                        vm.saveUserProfile(ageVal, weightVal, heightVal, genderVal)
                        editing = false
                    }) { Text("Save") }
                    Button(onClick = {
                        ageText = if (age > 0) age.toString() else ""
                        weightText = if (weight > 0.0) String.format("%.1f", weight) else ""
                        heightText = if (height > 0.0) String.format("%.1f", height) else ""
                        // reset genderIndex to stored value
                        val idx = genderOptions.indexOfFirst { it.equals(gender, ignoreCase = true) }
                        genderIndex = if (idx >= 0) idx else 0
                        editing = false
                    }) { Text("Cancel") }
                }
            }

            // Thresholds removed; CNN model handles classification

            // Danger zone: Reset data
            Text("Data Management")
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = { vm.resetAllData() }) { Text("Reset Data") }
            }
        }
    }
}

// Threshold settings removed
