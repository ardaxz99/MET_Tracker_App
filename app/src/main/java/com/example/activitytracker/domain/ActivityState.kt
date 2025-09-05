package com.example.activitytracker.domain

enum class ActivityState { Sedentary, Small, Medium, Vigorous }

val ActivityState.displayName: String
    get() = when (this) {
        ActivityState.Sedentary -> "Sedentary"
        ActivityState.Small -> "Light"
        ActivityState.Medium -> "Moderate"
        ActivityState.Vigorous -> "Vigorous"
    }
