@file:OptIn(ExperimentalMaterial3Api::class)

package com.example.activitytracker.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.example.activitytracker.R
import com.example.activitytracker.domain.ActivityState
import com.example.activitytracker.domain.displayName
import com.example.activitytracker.viewmodel.MainViewModel

@Composable
fun MainScreen(vm: MainViewModel, onHistory: () -> Unit, onSettings: () -> Unit) {
    val state by vm.activityState.collectAsState()
    val imuEnabled by vm.imuEnabled.collectAsState()
    val live by vm.liveAccel.collectAsState()

    Scaffold(
        topBar = { TopAppBar(title = { Text("Activity Monitor") }) }
    ) { padding ->
        Column(
            Modifier
                .padding(padding)
                .padding(16.dp)
                .fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Text("Current State", style = MaterialTheme.typography.headlineSmall)
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(20.dp)) {
                        val icon = when (state) {
                            ActivityState.Sedentary -> R.drawable.ic_state_sedentary
                            ActivityState.Small -> R.drawable.ic_state_light
                            ActivityState.Medium -> R.drawable.ic_state_moderate
                            ActivityState.Vigorous -> R.drawable.ic_state_vigorous
                        }
                        Icon(
                            painter = painterResource(id = icon),
                            contentDescription = null,
                            modifier = Modifier.size(72.dp)
                        )
                        Text(state.displayName, style = MaterialTheme.typography.headlineMedium)
                    }
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Debug Information")
                Switch(checked = imuEnabled, onCheckedChange = { vm.setImuEnabled(it) })
            }

            if (imuEnabled) {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Accelerometer (x, y, z)", style = MaterialTheme.typography.titleSmall)
                        val triple = live
                        Text(text = "${triple.first.format(2)}, ${triple.second.format(2)}, ${triple.third.format(2)}")
                        val rate by vm.samplingRateHz.collectAsState()
                        val w by vm.windowSeconds.collectAsState()
                        val s by vm.strideSeconds.collectAsState()
                        Text(text = "Sampling: ${rate.format(1)} Hz")
                        val probs by vm.cnnProbs.collectAsState()
                        if (probs.size >= 4) {
                            Text(text = "CNN probs: ${probs[0].format(2)}, ${probs[1].format(2)}, ${probs[2].format(2)}, ${probs[3].format(2)}")
                        }
                        val cnt by vm.windowSamples.collectAsState()
                        val fill by vm.windowFillSamples.collectAsState()
                        val samplesLabel = if (fill == cnt || cnt <= 0) "Samples/window: ${cnt}" else "Samples/window: ${fill}/${cnt}"
                        Text(text = samplesLabel)
                        Text(text = "Window: ${w.format(1)} s  •  Stride: ${s.format(1)} s")
                    }
                }
            }

            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = onHistory) { Text("History") }
                Button(onClick = onSettings) { Text("Settings") }
            }
        }
    }
}

private fun Float.format(digits: Int) = "% .${digits}f".replace(" ", "").format(this)
private fun Double.format(digits: Int) = "% .${digits}f".replace(" ", "").format(this)
