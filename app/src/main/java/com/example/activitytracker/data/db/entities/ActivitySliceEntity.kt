package com.example.activitytracker.data.db.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "activity_slices")
data class ActivitySliceEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: Long,
    val start: Long,
    val end: Long?,
    val state: String
)

