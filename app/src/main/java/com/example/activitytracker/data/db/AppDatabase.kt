package com.example.activitytracker.data.db

import androidx.room.Database
import androidx.room.RoomDatabase
import com.example.activitytracker.data.db.dao.SessionDao
import com.example.activitytracker.data.db.dao.SliceDao
import com.example.activitytracker.data.db.entities.ActivitySessionEntity
import com.example.activitytracker.data.db.entities.ActivitySliceEntity

@Database(
    entities = [ActivitySessionEntity::class, ActivitySliceEntity::class, com.example.activitytracker.data.db.entities.ActivitySecondEntity::class],
    version = 2,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun sessionDao(): SessionDao
    abstract fun sliceDao(): SliceDao
    abstract fun secondDao(): com.example.activitytracker.data.db.dao.SecondDao
}
