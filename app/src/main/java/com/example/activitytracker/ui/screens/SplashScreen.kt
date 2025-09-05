package com.example.activitytracker.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.dp
import com.example.activitytracker.R
import com.example.activitytracker.service.TrackingService

@Composable
fun SplashScreen(onStart: () -> Unit) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(verticalArrangement = Arrangement.spacedBy(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Image(
                painter = painterResource(id = R.drawable.ic_running_man),
                contentDescription = "Running icon",
                modifier = Modifier.height(140.dp)
            )
            Text(
                text = "Activity State Tracker",
                style = MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.SemiBold, letterSpacing = 0.5.sp),
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onBackground
            )
            Spacer(Modifier.height(4.dp))
            val ctx = LocalContext.current
            Button(
                onClick = {
                    // Start foreground service so tracking continues in background
                    val intent = android.content.Intent(ctx, TrackingService::class.java)
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                        ctx.startForegroundService(intent)
                    } else {
                        ctx.startService(intent)
                    }
                    onStart()
                },
                modifier = Modifier.height(56.dp),
                colors = ButtonDefaults.buttonColors()
            ) {
                Text("Start", style = MaterialTheme.typography.titleLarge)
            }
        }
    }
}
