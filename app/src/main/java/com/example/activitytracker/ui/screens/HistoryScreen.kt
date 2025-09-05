@file:OptIn(ExperimentalMaterial3Api::class)

package com.example.activitytracker.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.example.activitytracker.R
import com.example.activitytracker.viewmodel.MainViewModel
import java.util.Calendar

@Composable
fun HistoryScreen(onBack: () -> Unit, vm: MainViewModel) {
    val tabs = listOf("Day", "Week", "Month", "Year")
    val selected = remember { mutableIntStateOf(0) }

    Scaffold(topBar = {
        TopAppBar(title = { Text("History") }, navigationIcon = {
            IconButton(onClick = onBack) { Text("Back") }
        })
    }) { padding ->
        Column(
            Modifier
                .padding(padding)
                .padding(16.dp)
                .fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            TabRow(selectedTabIndex = selected.intValue) {
                tabs.forEachIndexed { index, title ->
                    Tab(
                        selected = selected.intValue == index,
                        onClick = { selected.intValue = index },
                        text = { Text(title) }
                    )
                }
            }

            when (selected.intValue) {
                0 -> DayHistory(vm)
                1 -> WeekHistory(vm)
                2 -> MonthHistory(vm)
                3 -> YearHistory(vm)
            }
        }
    }
}

@Composable
private fun DayHistory(vm: MainViewModel) {
    val cal = remember {
        Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
    }
    val dayStartSec = cal.timeInMillis / 1000L
    val counts by vm.repoDayHourCounts(dayStartSec).collectAsState(initial = emptyList())
    val age by vm.ageYears.collectAsState()
    val weight by vm.weightKg.collectAsState()
    val height by vm.heightCm.collectAsState()
    val gender by vm.gender.collectAsState()

    val states = listOf("Sedentary", "Light", "Moderate", "Vigorous")
    val matrix = Array(4) { DoubleArray(24) { 0.0 } }
    val totals = DoubleArray(4) { 0.0 }
    counts.forEach { hsc ->
        val row = states.indexOf(hsc.state).coerceAtLeast(0)
        val hour = hsc.hour.coerceIn(0, 23)
        matrix[row][hour] = (hsc.cnt.toDouble() / 3600.0).coerceIn(0.0, 1.0)
        totals[row] += hsc.cnt.toDouble()
    }
    val totalSec = totals.sum().coerceAtLeast(0.0)
    val fractions = if (totalSec > 0.0) totals.map { it / totalSec } else listOf(0.0, 0.0, 0.0, 0.0)

    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        // Summary pie chart for the day
        if (totalSec > 0.0) {
            Text("Today Summary", style = MaterialTheme.typography.titleMedium)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                PieChart(fractions = fractions, size = 160.dp)
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    val colors = stateColors()
                    for (i in 0 until 4) {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            ColorSwatch(colors[i])
                            val pct = (fractions[i] * 100.0)
                            Text("${states[i]} ${"%.1f".format(pct)}%", style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
            }
            // Calorie summary (net, excludes resting)
            val met = doubleArrayOf(1.2, 2.5, 4.5, 8.0)
            val sConst = if (gender.equals("male", ignoreCase = true)) 5.0 else -161.0
            val profileOk = age > 0 && weight > 0.0 && height > 0.0
            if (profileOk) {
                val rmr = 10.0 * weight + 6.25 * height - 5.0 * age + sConst
                val restPerSec = rmr / (24.0 * 3600.0)
                var net = 0.0
                for (i in 0 until 4) {
                    val seconds = totals[i]
                    val factor = (met[i] - 1.0).coerceAtLeast(0.0)
                    net += seconds * factor * restPerSec
                }
                Text("Calories today (net): ${"%.0f".format(net)} kcal", style = MaterialTheme.typography.titleMedium)
            } else {
                Text("Set age/weight/height in Settings to see calories.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            }
        }
        val icons = listOf(
            R.drawable.ic_state_sedentary,
            R.drawable.ic_state_light,
            R.drawable.ic_state_moderate,
            R.drawable.ic_state_vigorous
        )
        // Render 4 groups: [logo+name aligned right], then [24-hour bar chart] below
        for (row in 0 until 4) {
            Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Icon(
                            painter = painterResource(id = icons[row]),
                            contentDescription = null,
                            modifier = Modifier.height(24.dp)
                        )
                        Text(states[row], style = MaterialTheme.typography.bodyMedium)
                    }
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                    for (hour in 0 until 24) {
                        HourBar(fraction = matrix[row][hour], height = 36.dp, modifier = Modifier.weight(1f))
                    }
                }
            }
        }
    }
}

@Composable
private fun stateColors(): List<Color> = listOf(
    // Distinct, high-contrast palette: Blue, Green, Orange, Magenta
    Color(0xFF1E88E5), // Sedentary
    Color(0xFF43A047), // Light
    Color(0xFFFB8C00), // Moderate
    Color(0xFFD81B60)  // Vigorous
)

@Composable
private fun PieChart(fractions: List<Double>, size: Dp, strokeWidth: Dp = 0.dp) {
    val colors = stateColors()
    Box(Modifier.width(size).height(size)) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            var startAngle = -90f
            fractions.forEachIndexed { idx, f ->
                val sweep = (f.coerceIn(0.0, 1.0) * 360f).toFloat()
                if (sweep > 0f) {
                    drawArc(
                        color = colors[idx],
                        startAngle = startAngle,
                        sweepAngle = sweep,
                        useCenter = true
                    )
                }
                startAngle += sweep
            }
        }
    }
}

@Composable
private fun ColorSwatch(color: Color) {
    Canvas(modifier = Modifier.width(14.dp).height(14.dp)) {
        drawRoundRect(color = color)
    }
}

@Composable
private fun WeekHistory(vm: MainViewModel) {
    val cal = remember {
        Calendar.getInstance().apply {
            firstDayOfWeek = Calendar.MONDAY
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
            // roll back to Monday
            while (get(Calendar.DAY_OF_WEEK) != Calendar.MONDAY) add(Calendar.DAY_OF_MONTH, -1)
        }
    }
    val weekStart = cal.timeInMillis / 1000L
    val counts by vm.repoWeekCounts(weekStart).collectAsState(initial = emptyList())

    val states = listOf("Sedentary", "Light", "Moderate", "Vigorous")
    val matrix = Array(4) { DoubleArray(7) { 0.0 } }
    counts.forEach { dsc ->
        val row = states.indexOf(dsc.state).coerceAtLeast(0)
        val idx = dsc.dayIndex.coerceIn(0, 6)
        matrix[row][idx] = (dsc.cnt.toDouble() / 86400.0).coerceIn(0.0, 1.0)
    }
    StateCharts(states = states, matrix = matrix, segments = 7)
}

@Composable
private fun MonthHistory(vm: MainViewModel) {
    val cal = remember {
        Calendar.getInstance().apply {
            set(Calendar.DAY_OF_MONTH, 1)
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
    }
    val daysInMonth = cal.getActualMaximum(Calendar.DAY_OF_MONTH)
    val monthStart = cal.timeInMillis / 1000L
    val counts by vm.repoMonthCounts(monthStart, daysInMonth).collectAsState(initial = emptyList())

    val states = listOf("Sedentary", "Light", "Moderate", "Vigorous")
    val matrix = Array(4) { DoubleArray(daysInMonth) { 0.0 } }
    counts.forEach { dsc ->
        val row = states.indexOf(dsc.state).coerceAtLeast(0)
        val idx = dsc.dayIndex.coerceIn(0, daysInMonth - 1)
        matrix[row][idx] = (dsc.cnt.toDouble() / 86400.0).coerceIn(0.0, 1.0)
    }
    StateCharts(states = states, matrix = matrix, segments = daysInMonth)
}

@Composable
private fun YearHistory(vm: MainViewModel) {
    val cal = remember {
        Calendar.getInstance().apply {
            set(Calendar.MONTH, Calendar.JANUARY)
            set(Calendar.DAY_OF_MONTH, 1)
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
    }
    val yearStart = cal.timeInMillis / 1000L
    val counts by vm.repoYearCounts(yearStart).collectAsState(initial = emptyList())

    val states = listOf("Sedentary", "Light", "Moderate", "Vigorous")
    val matrix = Array(4) { DoubleArray(12) { 0.0 } }
    // Normalize by days per month for a fair fraction per month
    val daysIn = IntArray(12) { i ->
        Calendar.getInstance().apply {
            set(Calendar.MONTH, i)
            set(Calendar.DAY_OF_MONTH, 1)
        }.getActualMaximum(Calendar.DAY_OF_MONTH)
    }
    counts.forEach { msc ->
        val row = states.indexOf(msc.state).coerceAtLeast(0)
        val idx = msc.monthIndex.coerceIn(0, 11)
        val denom = (daysIn[idx] * 86400.0)
        matrix[row][idx] = (msc.cnt.toDouble() / denom).coerceIn(0.0, 1.0)
    }
    StateCharts(states = states, matrix = matrix, segments = 12)
}

@Composable
private fun StateCharts(states: List<String>, matrix: Array<DoubleArray>, segments: Int) {
    val icons = listOf(
        R.drawable.ic_state_sedentary,
        R.drawable.ic_state_light,
        R.drawable.ic_state_moderate,
        R.drawable.ic_state_vigorous
    )
    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        for (row in 0 until 4) {
            Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Icon(
                            painter = painterResource(id = icons[row]),
                            contentDescription = null,
                            modifier = Modifier.height(24.dp)
                        )
                        Text(states[row], style = MaterialTheme.typography.bodyMedium)
                    }
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                    for (i in 0 until segments) {
                        HourBar(fraction = matrix[row][i], height = 36.dp, modifier = Modifier.weight(1f))
                    }
                }
            }
        }
    }
}

@Composable
private fun HourBar(fraction: Double, height: Dp, modifier: Modifier) {
    val clamped = fraction.coerceIn(0.0, 1.0)
    val barColor = MaterialTheme.colorScheme.primary
    Box(modifier.height(height)) {
        // Background
        Canvas(modifier = Modifier.fillMaxSize()) {
            drawRoundRect(color = Color.DarkGray.copy(alpha = 0.3f))
        }
        // Foreground fill from bottom
        Canvas(modifier = Modifier.fillMaxSize()) {
            val h = size.height * clamped.toFloat()
            drawRect(
                color = barColor,
                topLeft = androidx.compose.ui.geometry.Offset(0f, size.height - h),
                size = androidx.compose.ui.geometry.Size(size.width, h)
            )
        }
    }
}
