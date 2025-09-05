package com.example.activitytracker.data.db.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Stores the predicted activity state for each elapsed epoch second.
 * Primary key is the epoch second to allow idempotent upserts per-second.
 */
@Entity(tableName = "activity_seconds")
data class ActivitySecondEntity(
    @PrimaryKey val tsSec: Long,
    val state: String
)

