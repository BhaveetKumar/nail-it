# Mobile App Integration

## Table of Contents

1. [Overview](#overview)
2. [Mobile App Architecture](#mobile-app-architecture)
3. [Core Features](#core-features)
4. [Offline Support](#offline-support)
5. [Push Notifications](#push-notifications)
6. [Cross-Platform Development](#cross-platform-development)
7. [API Integration](#api-integration)
8. [Follow-up Questions](#follow-up-questions)
9. [Sources](#sources)

## Overview

### Learning Objectives

- Develop mobile applications for the Master Engineer Curriculum
- Implement cross-platform mobile solutions
- Integrate with backend services and APIs
- Provide offline learning capabilities
- Create engaging mobile learning experiences

### What is Mobile App Integration?

Mobile app integration involves creating mobile applications that provide access to the Master Engineer Curriculum on smartphones and tablets, with features like offline learning, progress tracking, and push notifications.

## Mobile App Architecture

### 1. Flutter Implementation

#### Flutter App Structure
```dart
// main.dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'screens/home_screen.dart';
import 'screens/lesson_screen.dart';
import 'screens/progress_screen.dart';
import 'providers/study_provider.dart';
import 'providers/auth_provider.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AuthProvider()),
        ChangeNotifierProvider(create: (_) => StudyProvider()),
      ],
      child: MaterialApp(
        title: 'Master Engineer Curriculum',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          visualDensity: VisualDensity.adaptivePlatformDensity,
        ),
        home: HomeScreen(),
        routes: {
          '/lesson': (context) => LessonScreen(),
          '/progress': (context) => ProgressScreen(),
        },
      ),
    );
  }
}

// models/lesson.dart
class Lesson {
  final String id;
  final String title;
  final String description;
  final String content;
  final String phaseId;
  final String moduleId;
  final int duration;
  final List<String> prerequisites;
  final String difficulty;
  final bool isCompleted;
  final double progress;

  Lesson({
    required this.id,
    required this.title,
    required this.description,
    required this.content,
    required this.phaseId,
    required this.moduleId,
    required this.duration,
    required this.prerequisites,
    required this.difficulty,
    this.isCompleted = false,
    this.progress = 0.0,
  });

  factory Lesson.fromJson(Map<String, dynamic> json) {
    return Lesson(
      id: json['id'],
      title: json['title'],
      description: json['description'],
      content: json['content'],
      phaseId: json['phaseId'],
      moduleId: json['moduleId'],
      duration: json['duration'],
      prerequisites: List<String>.from(json['prerequisites']),
      difficulty: json['difficulty'],
      isCompleted: json['isCompleted'] ?? false,
      progress: json['progress']?.toDouble() ?? 0.0,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'title': title,
      'description': description,
      'content': content,
      'phaseId': phaseId,
      'moduleId': moduleId,
      'duration': duration,
      'prerequisites': prerequisites,
      'difficulty': difficulty,
      'isCompleted': isCompleted,
      'progress': progress,
    };
  }
}

// models/study_session.dart
class StudySession {
  final String id;
  final String lessonId;
  final DateTime startTime;
  final DateTime endTime;
  final Duration duration;
  final int rating;
  final String notes;
  final bool isCompleted;

  StudySession({
    required this.id,
    required this.lessonId,
    required this.startTime,
    required this.endTime,
    required this.duration,
    required this.rating,
    required this.notes,
    this.isCompleted = false,
  });

  factory StudySession.fromJson(Map<String, dynamic> json) {
    return StudySession(
      id: json['id'],
      lessonId: json['lessonId'],
      startTime: DateTime.parse(json['startTime']),
      endTime: DateTime.parse(json['endTime']),
      duration: Duration(seconds: json['duration']),
      rating: json['rating'],
      notes: json['notes'],
      isCompleted: json['isCompleted'] ?? false,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'lessonId': lessonId,
      'startTime': startTime.toIso8601String(),
      'endTime': endTime.toIso8601String(),
      'duration': duration.inSeconds,
      'rating': rating,
      'notes': notes,
      'isCompleted': isCompleted,
    };
  }
}

// providers/study_provider.dart
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import '../models/lesson.dart';
import '../models/study_session.dart';
import '../services/api_service.dart';
import '../services/local_storage_service.dart';

class StudyProvider with ChangeNotifier {
  List<Lesson> _lessons = [];
  List<StudySession> _sessions = [];
  bool _isLoading = false;
  String _currentPhase = 'phase0';
  double _overallProgress = 0.0;

  List<Lesson> get lessons => _lessons;
  List<StudySession> get sessions => _sessions;
  bool get isLoading => _isLoading;
  String get currentPhase => _currentPhase;
  double get overallProgress => _overallProgress;

  final ApiService _apiService = ApiService();
  final LocalStorageService _localStorage = LocalStorageService();

  Future<void> loadLessons() async {
    _isLoading = true;
    notifyListeners();

    try {
      // Try to load from API first
      _lessons = await _apiService.getLessons();
      await _localStorage.saveLessons(_lessons);
    } catch (e) {
      // Fallback to local storage
      _lessons = await _localStorage.getLessons();
    }

    _calculateOverallProgress();
    _isLoading = false;
    notifyListeners();
  }

  Future<void> loadSessions() async {
    _sessions = await _localStorage.getSessions();
    notifyListeners();
  }

  Future<void> startLesson(String lessonId) async {
    final lesson = _lessons.firstWhere((l) => l.id == lessonId);
    final session = StudySession(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      lessonId: lessonId,
      startTime: DateTime.now(),
      endTime: DateTime.now(),
      duration: Duration.zero,
      rating: 0,
      notes: '',
    );

    _sessions.add(session);
    await _localStorage.saveSessions(_sessions);
    notifyListeners();
  }

  Future<void> completeLesson(String lessonId, int rating, String notes) async {
    final sessionIndex = _sessions.indexWhere((s) => s.lessonId == lessonId && !s.isCompleted);
    if (sessionIndex != -1) {
      final session = _sessions[sessionIndex];
      final completedSession = StudySession(
        id: session.id,
        lessonId: session.lessonId,
        startTime: session.startTime,
        endTime: DateTime.now(),
        duration: DateTime.now().difference(session.startTime),
        rating: rating,
        notes: notes,
        isCompleted: true,
      );

      _sessions[sessionIndex] = completedSession;
      await _localStorage.saveSessions(_sessions);

      // Update lesson progress
      final lessonIndex = _lessons.indexWhere((l) => l.id == lessonId);
      if (lessonIndex != -1) {
        _lessons[lessonIndex] = Lesson(
          id: _lessons[lessonIndex].id,
          title: _lessons[lessonIndex].title,
          description: _lessons[lessonIndex].description,
          content: _lessons[lessonIndex].content,
          phaseId: _lessons[lessonIndex].phaseId,
          moduleId: _lessons[lessonIndex].moduleId,
          duration: _lessons[lessonIndex].duration,
          prerequisites: _lessons[lessonIndex].prerequisites,
          difficulty: _lessons[lessonIndex].difficulty,
          isCompleted: true,
          progress: 100.0,
        );
      }

      _calculateOverallProgress();
      notifyListeners();
    }
  }

  void _calculateOverallProgress() {
    if (_lessons.isEmpty) {
      _overallProgress = 0.0;
      return;
    }

    final completedLessons = _lessons.where((l) => l.isCompleted).length;
    _overallProgress = (completedLessons / _lessons.length) * 100;
  }

  List<Lesson> getLessonsByPhase(String phaseId) {
    return _lessons.where((l) => l.phaseId == phaseId).toList();
  }

  List<Lesson> getCompletedLessons() {
    return _lessons.where((l) => l.isCompleted).toList();
  }

  List<StudySession> getSessionsByLesson(String lessonId) {
    return _sessions.where((s) => s.lessonId == lessonId).toList();
  }

  Duration getTotalStudyTime() {
    return _sessions.fold(Duration.zero, (total, session) => total + session.duration);
  }

  int getStudyStreak() {
    if (_sessions.isEmpty) return 0;

    final sortedSessions = List<StudySession>.from(_sessions)
      ..sort((a, b) => a.startTime.compareTo(b.startTime));

    int streak = 0;
    DateTime currentDate = DateTime.now().toUtc().subtract(Duration(days: 1));

    for (int i = sortedSessions.length - 1; i >= 0; i--) {
      final sessionDate = sortedSessions[i].startTime.toUtc();
      if (sessionDate.day == currentDate.day &&
          sessionDate.month == currentDate.month &&
          sessionDate.year == currentDate.year) {
        streak++;
        currentDate = currentDate.subtract(Duration(days: 1));
      } else {
        break;
      }
    }

    return streak;
  }
}

// screens/home_screen.dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/study_provider.dart';
import '../widgets/phase_card.dart';
import '../widgets/progress_chart.dart';
import '../widgets/study_stats.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Provider.of<StudyProvider>(context, listen: false).loadLessons();
      Provider.of<StudyProvider>(context, listen: false).loadSessions();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Master Engineer Curriculum'),
        actions: [
          IconButton(
            icon: Icon(Icons.analytics),
            onPressed: () => Navigator.pushNamed(context, '/progress'),
          ),
        ],
      ),
      body: Consumer<StudyProvider>(
        builder: (context, studyProvider, child) {
          if (studyProvider.isLoading) {
            return Center(child: CircularProgressIndicator());
          }

          return SingleChildScrollView(
            padding: EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Progress Overview
                ProgressChart(
                  progress: studyProvider.overallProgress,
                  totalLessons: studyProvider.lessons.length,
                  completedLessons: studyProvider.getCompletedLessons().length,
                ),
                SizedBox(height: 20),

                // Study Stats
                StudyStats(
                  totalStudyTime: studyProvider.getTotalStudyTime(),
                  studyStreak: studyProvider.getStudyStreak(),
                  totalSessions: studyProvider.sessions.length,
                ),
                SizedBox(height: 20),

                // Phase Cards
                Text(
                  'Learning Phases',
                  style: Theme.of(context).textTheme.headline5,
                ),
                SizedBox(height: 10),
                _buildPhaseCards(studyProvider),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildPhaseCards(StudyProvider studyProvider) {
    final phases = [
      {'id': 'phase0', 'name': 'Phase 0: Fundamentals', 'color': Colors.green},
      {'id': 'phase1', 'name': 'Phase 1: Intermediate', 'color': Colors.blue},
      {'id': 'phase2', 'name': 'Phase 2: Advanced', 'color': Colors.orange},
      {'id': 'phase3', 'name': 'Phase 3: Expert', 'color': Colors.purple},
    ];

    return Column(
      children: phases.map((phase) {
        final phaseLessons = studyProvider.getLessonsByPhase(phase['id'] as String);
        final completedLessons = phaseLessons.where((l) => l.isCompleted).length;
        final progress = phaseLessons.isEmpty ? 0.0 : (completedLessons / phaseLessons.length) * 100;

        return PhaseCard(
          title: phase['name'] as String,
          progress: progress,
          totalLessons: phaseLessons.length,
          completedLessons: completedLessons,
          color: phase['color'] as Color,
          onTap: () => _navigateToPhase(phase['id'] as String),
        );
      }).toList(),
    );
  }

  void _navigateToPhase(String phaseId) {
    // Navigate to phase details screen
    // Implementation depends on your navigation structure
  }
}

// widgets/progress_chart.dart
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

class ProgressChart extends StatelessWidget {
  final double progress;
  final int totalLessons;
  final int completedLessons;

  const ProgressChart({
    Key? key,
    required this.progress,
    required this.totalLessons,
    required this.completedLessons,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Overall Progress',
              style: Theme.of(context).textTheme.headline6,
            ),
            SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '${progress.toStringAsFixed(1)}%',
                        style: Theme.of(context).textTheme.headline4?.copyWith(
                          color: Theme.of(context).primaryColor,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        '$completedLessons of $totalLessons lessons completed',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ],
                  ),
                ),
                SizedBox(
                  width: 100,
                  height: 100,
                  child: CircularProgressIndicator(
                    value: progress / 100,
                    strokeWidth: 8,
                    backgroundColor: Colors.grey[300],
                    valueColor: AlwaysStoppedAnimation<Color>(
                      Theme.of(context).primaryColor,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// widgets/study_stats.dart
import 'package:flutter/material.dart';

class StudyStats extends StatelessWidget {
  final Duration totalStudyTime;
  final int studyStreak;
  final int totalSessions;

  const StudyStats({
    Key? key,
    required this.totalStudyTime,
    required this.studyStreak,
    required this.totalSessions,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Study Statistics',
              style: Theme.of(context).textTheme.headline6,
            ),
            SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStatItem(
                  context,
                  'Total Time',
                  _formatDuration(totalStudyTime),
                  Icons.access_time,
                ),
                _buildStatItem(
                  context,
                  'Current Streak',
                  '$studyStreak days',
                  Icons.local_fire_department,
                ),
                _buildStatItem(
                  context,
                  'Sessions',
                  '$totalSessions',
                  Icons.play_lesson,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(BuildContext context, String label, String value, IconData icon) {
    return Column(
      children: [
        Icon(icon, size: 32, color: Theme.of(context).primaryColor),
        SizedBox(height: 8),
        Text(
          value,
          style: Theme.of(context).textTheme.headline6?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }

  String _formatDuration(Duration duration) {
    final hours = duration.inHours;
    final minutes = duration.inMinutes % 60;
    
    if (hours > 0) {
      return '${hours}h ${minutes}m';
    } else {
      return '${minutes}m';
    }
  }
}
```

### 2. React Native Implementation

#### React Native App Structure
```javascript
// App.js
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider } from 'react-redux';
import { store } from './src/store';
import HomeScreen from './src/screens/HomeScreen';
import LessonsScreen from './src/screens/LessonsScreen';
import ProgressScreen from './src/screens/ProgressScreen';
import ProfileScreen from './src/screens/ProfileScreen';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <Provider store={store}>
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={({ route }) => ({
            tabBarIcon: ({ focused, color, size }) => {
              let iconName;
              if (route.name === 'Home') {
                iconName = focused ? 'home' : 'home-outline';
              } else if (route.name === 'Lessons') {
                iconName = focused ? 'book' : 'book-outline';
              } else if (route.name === 'Progress') {
                iconName = focused ? 'analytics' : 'analytics-outline';
              } else if (route.name === 'Profile') {
                iconName = focused ? 'person' : 'person-outline';
              }
              return <Ionicons name={iconName} size={size} color={color} />;
            },
            tabBarActiveTintColor: '#007AFF',
            tabBarInactiveTintColor: 'gray',
          })}
        >
          <Tab.Screen name="Home" component={HomeScreen} />
          <Tab.Screen name="Lessons" component={LessonsScreen} />
          <Tab.Screen name="Progress" component={ProgressScreen} />
          <Tab.Screen name="Profile" component={ProfileScreen} />
        </Tab.Navigator>
      </NavigationContainer>
    </Provider>
  );
}

// src/store/index.js
import { createStore, combineReducers, applyMiddleware } from 'redux';
import { persistStore, persistReducer } from 'redux-persist';
import AsyncStorage from '@react-native-async-storage/async-storage';
import thunk from 'redux-thunk';
import lessonsReducer from './reducers/lessonsReducer';
import progressReducer from './reducers/progressReducer';
import authReducer from './reducers/authReducer';

const persistConfig = {
  key: 'root',
  storage: AsyncStorage,
  whitelist: ['lessons', 'progress', 'auth'],
};

const rootReducer = combineReducers({
  lessons: lessonsReducer,
  progress: progressReducer,
  auth: authReducer,
});

const persistedReducer = persistReducer(persistConfig, rootReducer);

export const store = createStore(persistedReducer, applyMiddleware(thunk));
export const persistor = persistStore(store);

// src/reducers/lessonsReducer.js
const initialState = {
  lessons: [],
  currentLesson: null,
  isLoading: false,
  error: null,
};

const lessonsReducer = (state = initialState, action) => {
  switch (action.type) {
    case 'FETCH_LESSONS_REQUEST':
      return {
        ...state,
        isLoading: true,
        error: null,
      };
    case 'FETCH_LESSONS_SUCCESS':
      return {
        ...state,
        lessons: action.payload,
        isLoading: false,
      };
    case 'FETCH_LESSONS_FAILURE':
      return {
        ...state,
        isLoading: false,
        error: action.payload,
      };
    case 'SET_CURRENT_LESSON':
      return {
        ...state,
        currentLesson: action.payload,
      };
    case 'UPDATE_LESSON_PROGRESS':
      return {
        ...state,
        lessons: state.lessons.map(lesson =>
          lesson.id === action.payload.lessonId
            ? { ...lesson, progress: action.payload.progress, isCompleted: action.payload.isCompleted }
            : lesson
        ),
      };
    default:
      return state;
  }
};

export default lessonsReducer;

// src/actions/lessonsActions.js
export const fetchLessons = () => async (dispatch) => {
  dispatch({ type: 'FETCH_LESSONS_REQUEST' });
  
  try {
    const response = await fetch('/api/lessons');
    const lessons = await response.json();
    dispatch({ type: 'FETCH_LESSONS_SUCCESS', payload: lessons });
  } catch (error) {
    dispatch({ type: 'FETCH_LESSONS_FAILURE', payload: error.message });
  }
};

export const updateLessonProgress = (lessonId, progress, isCompleted) => ({
  type: 'UPDATE_LESSON_PROGRESS',
  payload: { lessonId, progress, isCompleted },
});

// src/screens/HomeScreen.js
import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import { fetchLessons } from '../actions/lessonsActions';
import ProgressChart from '../components/ProgressChart';
import StudyStats from '../components/StudyStats';
import PhaseCard from '../components/PhaseCard';

const HomeScreen = ({ navigation }) => {
  const dispatch = useDispatch();
  const { lessons, isLoading } = useSelector(state => state.lessons);
  const { overallProgress, totalStudyTime, studyStreak } = useSelector(state => state.progress);

  useEffect(() => {
    dispatch(fetchLessons());
  }, [dispatch]);

  const phases = [
    { id: 'phase0', name: 'Phase 0: Fundamentals', color: '#4CAF50' },
    { id: 'phase1', name: 'Phase 1: Intermediate', color: '#2196F3' },
    { id: 'phase2', name: 'Phase 2: Advanced', color: '#FF9800' },
    { id: 'phase3', name: 'Phase 3: Expert', color: '#9C27B0' },
  ];

  const getPhaseProgress = (phaseId) => {
    const phaseLessons = lessons.filter(lesson => lesson.phaseId === phaseId);
    const completedLessons = phaseLessons.filter(lesson => lesson.isCompleted);
    return phaseLessons.length > 0 ? (completedLessons.length / phaseLessons.length) * 100 : 0;
  };

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <Text>Loading...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Master Engineer Curriculum</Text>
        <Text style={styles.subtitle}>Your learning journey starts here</Text>
      </View>

      <ProgressChart
        progress={overallProgress}
        totalLessons={lessons.length}
        completedLessons={lessons.filter(lesson => lesson.isCompleted).length}
      />

      <StudyStats
        totalStudyTime={totalStudyTime}
        studyStreak={studyStreak}
        totalSessions={lessons.length}
      />

      <View style={styles.phasesContainer}>
        <Text style={styles.sectionTitle}>Learning Phases</Text>
        {phases.map(phase => (
          <PhaseCard
            key={phase.id}
            title={phase.name}
            progress={getPhaseProgress(phase.id)}
            color={phase.color}
            onPress={() => navigation.navigate('Lessons', { phaseId: phase.id })}
          />
        ))}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    padding: 20,
    backgroundColor: '#fff',
    marginBottom: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginTop: 5,
  },
  phasesContainer: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
});

export default HomeScreen;
```

## Core Features

### 1. Offline Learning

#### Offline Content Management
```dart
// services/offline_service.dart
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

class OfflineService {
  static const String _lessonsKey = 'offline_lessons';
  static const String _sessionsKey = 'offline_sessions';
  static const String _progressKey = 'offline_progress';

  Future<void> downloadContentForOffline() async {
    // Download lessons content
    final lessons = await _downloadLessons();
    await _saveLessonsOffline(lessons);

    // Download media files
    await _downloadMediaFiles(lessons);

    // Download code examples
    await _downloadCodeExamples(lessons);
  }

  Future<List<Lesson>> _downloadLessons() async {
    // Implementation to download lessons from API
    // This would typically involve making HTTP requests
    return [];
  }

  Future<void> _saveLessonsOffline(List<Lesson> lessons) async {
    final prefs = await SharedPreferences.getInstance();
    final lessonsJson = lessons.map((lesson) => lesson.toJson()).toList();
    await prefs.setString(_lessonsKey, jsonEncode(lessonsJson));
  }

  Future<List<Lesson>> getOfflineLessons() async {
    final prefs = await SharedPreferences.getInstance();
    final lessonsJson = prefs.getString(_lessonsKey);
    
    if (lessonsJson != null) {
      final List<dynamic> decoded = jsonDecode(lessonsJson);
      return decoded.map((json) => Lesson.fromJson(json)).toList();
    }
    
    return [];
  }

  Future<void> _downloadMediaFiles(List<Lesson> lessons) async {
    final directory = await getApplicationDocumentsDirectory();
    final mediaDir = Directory('${directory.path}/media');
    
    if (!await mediaDir.exists()) {
      await mediaDir.create(recursive: true);
    }

    for (final lesson in lessons) {
      for (final mediaUrl in lesson.mediaUrls) {
        await _downloadFile(mediaUrl, '${mediaDir.path}/${_getFileName(mediaUrl)}');
      }
    }
  }

  Future<void> _downloadFile(String url, String filePath) async {
    // Implementation to download file from URL
    // This would typically involve making HTTP requests and saving to file
  }

  String _getFileName(String url) {
    return url.split('/').last;
  }

  Future<bool> isContentAvailableOffline() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.containsKey(_lessonsKey);
  }

  Future<void> syncOfflineData() async {
    // Sync offline data when connection is available
    final offlineSessions = await getOfflineSessions();
    final offlineProgress = await getOfflineProgress();

    // Upload to server
    await _uploadSessions(offlineSessions);
    await _uploadProgress(offlineProgress);

    // Clear offline data after successful sync
    await _clearOfflineData();
  }

  Future<List<StudySession>> getOfflineSessions() async {
    final prefs = await SharedPreferences.getInstance();
    final sessionsJson = prefs.getString(_sessionsKey);
    
    if (sessionsJson != null) {
      final List<dynamic> decoded = jsonDecode(sessionsJson);
      return decoded.map((json) => StudySession.fromJson(json)).toList();
    }
    
    return [];
  }

  Future<Map<String, dynamic>> getOfflineProgress() async {
    final prefs = await SharedPreferences.getInstance();
    final progressJson = prefs.getString(_progressKey);
    
    if (progressJson != null) {
      return jsonDecode(progressJson);
    }
    
    return {};
  }

  Future<void> _uploadSessions(List<StudySession> sessions) async {
    // Implementation to upload sessions to server
  }

  Future<void> _uploadProgress(Map<String, dynamic> progress) async {
    // Implementation to upload progress to server
  }

  Future<void> _clearOfflineData() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_sessionsKey);
    await prefs.remove(_progressKey);
  }
}
```

### 2. Push Notifications

#### Notification Service
```dart
// services/notification_service.dart
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:timezone/timezone.dart' as tz;
import 'package:timezone/data/latest.dart' as tz;

class NotificationService {
  static final FlutterLocalNotificationsPlugin _notifications = 
      FlutterLocalNotificationsPlugin();
  static final FirebaseMessaging _firebaseMessaging = FirebaseMessaging.instance;

  static Future<void> initialize() async {
    tz.initializeTimeZones();

    const AndroidInitializationSettings androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    
    const DarwinInitializationSettings iosSettings =
        DarwinInitializationSettings(
          requestAlertPermission: true,
          requestBadgePermission: true,
          requestSoundPermission: true,
        );

    const InitializationSettings settings = InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );

    await _notifications.initialize(settings);
    await _requestPermissions();
    await _setupFirebaseMessaging();
  }

  static Future<void> _requestPermissions() async {
    await _notifications
        .resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>()
        ?.requestNotificationsPermission();
  }

  static Future<void> _setupFirebaseMessaging() async {
    await _firebaseMessaging.requestPermission();
    
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      _showNotification(
        message.notification?.title ?? 'New Update',
        message.notification?.body ?? 'You have a new notification',
      );
    });
  }

  static Future<void> scheduleStudyReminder(DateTime time, String message) async {
    await _notifications.zonedSchedule(
      0,
      'Study Reminder',
      message,
      tz.TZDateTime.from(time, tz.local),
      const NotificationDetails(
        android: AndroidNotificationDetails(
          'study_reminders',
          'Study Reminders',
          channelDescription: 'Reminders for study sessions',
          importance: Importance.high,
          priority: Priority.high,
        ),
        iOS: DarwinNotificationDetails(),
      ),
      uiLocalNotificationDateInterpretation:
          UILocalNotificationDateInterpretation.absoluteTime,
      matchDateTimeComponents: DateTimeComponents.time,
    );
  }

  static Future<void> scheduleDailyReminder(TimeOfDay time, String message) async {
    final now = DateTime.now();
    var scheduledDate = DateTime(now.year, now.month, now.day, time.hour, time.minute);
    
    if (scheduledDate.isBefore(now)) {
      scheduledDate = scheduledDate.add(const Duration(days: 1));
    }

    await _notifications.zonedSchedule(
      1,
      'Daily Study Reminder',
      message,
      tz.TZDateTime.from(scheduledDate, tz.local),
      const NotificationDetails(
        android: AndroidNotificationDetails(
          'daily_reminders',
          'Daily Study Reminders',
          channelDescription: 'Daily reminders to study',
          importance: Importance.high,
          priority: Priority.high,
        ),
        iOS: DarwinNotificationDetails(),
      ),
      uiLocalNotificationDateInterpretation:
          UILocalNotificationDateInterpretation.absoluteTime,
      matchDateTimeComponents: DateTimeComponents.time,
    );
  }

  static Future<void> _showNotification(String title, String body) async {
    const AndroidNotificationDetails androidDetails = AndroidNotificationDetails(
      'general',
      'General Notifications',
      channelDescription: 'General notifications from the app',
      importance: Importance.high,
      priority: Priority.high,
    );

    const DarwinNotificationDetails iosDetails = DarwinNotificationDetails();

    const NotificationDetails details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _notifications.show(0, title, body, details);
  }

  static Future<void> cancelAllNotifications() async {
    await _notifications.cancelAll();
  }

  static Future<void> cancelNotification(int id) async {
    await _notifications.cancel(id);
  }
}
```

## Cross-Platform Development

### 1. Flutter vs React Native

#### Comparison Table
| Feature | Flutter | React Native |
|---------|---------|--------------|
| **Performance** | Native performance | Near-native performance |
| **Development Speed** | Fast with hot reload | Fast with fast refresh |
| **UI Components** | Custom widgets | Native components |
| **Platform Support** | iOS, Android, Web, Desktop | iOS, Android, Web |
| **Learning Curve** | Dart language | JavaScript/TypeScript |
| **Community** | Growing rapidly | Large and mature |
| **Code Sharing** | High (90%+) | High (80%+) |
| **Native Features** | Good | Excellent |
| **Size** | Larger app size | Smaller app size |

### 2. Platform-Specific Features

#### iOS Implementation
```swift
// iOS/Swift implementation for native features
import UIKit
import UserNotifications

class StudyReminderManager {
    static let shared = StudyReminderManager()
    
    func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            if granted {
                print("Notification permission granted")
            } else {
                print("Notification permission denied")
            }
        }
    }
    
    func scheduleStudyReminder(title: String, body: String, time: Date) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        
        let trigger = UNCalendarNotificationTrigger(dateMatching: Calendar.current.dateComponents([.hour, .minute], from: time), repeats: true)
        let request = UNNotificationRequest(identifier: "study_reminder", content: content, trigger: trigger)
        
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Error scheduling notification: \(error)")
            }
        }
    }
}
```

#### Android Implementation
```kotlin
// Android/Kotlin implementation for native features
class StudyReminderManager(private val context: Context) {
    
    fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ActivityCompat.requestPermissions(
                context as Activity,
                arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                1
            )
        }
    }
    
    fun scheduleStudyReminder(title: String, body: String, time: Calendar) {
        val intent = Intent(context, StudyReminderReceiver::class.java).apply {
            putExtra("title", title)
            putExtra("body", body)
        }
        
        val pendingIntent = PendingIntent.getBroadcast(
            context,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        alarmManager.setRepeating(
            AlarmManager.RTC_WAKEUP,
            time.timeInMillis,
            AlarmManager.INTERVAL_DAY,
            pendingIntent
        )
    }
}
```

## API Integration

### 1. REST API Client

#### API Service Implementation
```dart
// services/api_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = 'https://api.masterengineer.com';
  static const String apiVersion = 'v1';
  
  String get _apiUrl => '$baseUrl/$apiVersion';

  Future<List<Lesson>> getLessons() async {
    try {
      final response = await http.get(
        Uri.parse('$_apiUrl/lessons'),
        headers: await _getHeaders(),
      );

      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        return data.map((json) => Lesson.fromJson(json)).toList();
      } else {
        throw Exception('Failed to load lessons: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<Lesson> getLesson(String lessonId) async {
    try {
      final response = await http.get(
        Uri.parse('$_apiUrl/lessons/$lessonId'),
        headers: await _getHeaders(),
      );

      if (response.statusCode == 200) {
        return Lesson.fromJson(jsonDecode(response.body));
      } else {
        throw Exception('Failed to load lesson: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<void> updateLessonProgress(String lessonId, double progress, bool isCompleted) async {
    try {
      final response = await http.put(
        Uri.parse('$_apiUrl/lessons/$lessonId/progress'),
        headers: await _getHeaders(),
        body: jsonEncode({
          'progress': progress,
          'isCompleted': isCompleted,
        }),
      );

      if (response.statusCode != 200) {
        throw Exception('Failed to update progress: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<void> createStudySession(StudySession session) async {
    try {
      final response = await http.post(
        Uri.parse('$_apiUrl/sessions'),
        headers: await _getHeaders(),
        body: jsonEncode(session.toJson()),
      );

      if (response.statusCode != 201) {
        throw Exception('Failed to create session: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<List<StudySession>> getStudySessions() async {
    try {
      final response = await http.get(
        Uri.parse('$_apiUrl/sessions'),
        headers: await _getHeaders(),
      );

      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        return data.map((json) => StudySession.fromJson(json)).toList();
      } else {
        throw Exception('Failed to load sessions: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<Map<String, dynamic>> getProgress() async {
    try {
      final response = await http.get(
        Uri.parse('$_apiUrl/progress'),
        headers: await _getHeaders(),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to load progress: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<Map<String, String>> _getHeaders() async {
    // Get auth token from secure storage
    final token = await _getAuthToken();
    
    return {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer $token',
    };
  }

  Future<String> _getAuthToken() async {
    // Implementation to get auth token from secure storage
    // This would typically involve reading from SharedPreferences or secure storage
    return 'your_auth_token_here';
  }
}
```

## Follow-up Questions

### 1. Mobile Development
**Q: Which mobile framework should I choose for the curriculum app?**
A: Choose based on your team's expertise, performance requirements, and platform needs. Flutter offers better performance and UI consistency, while React Native has a larger community and easier native integration.

### 2. Offline Support
**Q: How do you implement effective offline learning?**
A: Download content when online, use local storage for progress tracking, implement sync mechanisms, and provide offline indicators to users.

### 3. Cross-Platform
**Q: What are the key considerations for cross-platform development?**
A: Consider performance requirements, native feature needs, team expertise, development timeline, and maintenance costs when choosing between Flutter and React Native.

## Sources

### Mobile Development
- **Flutter**: [Official Documentation](https://flutter.dev/docs)
- **React Native**: [Official Documentation](https://reactnative.dev/docs/getting-started)
- **Expo**: [React Native Platform](https://expo.dev/)

### Cross-Platform Tools
- **Xamarin**: [Microsoft's Mobile Platform](https://dotnet.microsoft.com/apps/xamarin)
- **Ionic**: [Hybrid Mobile Framework](https://ionicframework.com/)
- **Cordova**: [Mobile Development Framework](https://cordova.apache.org/)

### Mobile Testing
- **Appium**: [Mobile App Testing](http://appium.io/)
- **Detox**: [React Native Testing](https://github.com/wix/Detox)
- **Flutter Testing**: [Flutter Test Documentation](https://flutter.dev/docs/testing)

---

**Next**: [Study Tracker](../study_tracker/README.md) | **Previous**: [Learning Resources](../learning_resources/README.md) | **Up**: [Mobile App](../README.md)
