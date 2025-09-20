# Computational Geometry

## Overview

This module covers computational geometry algorithms including convex hull, line intersection, polygon operations, and spatial data structures. These concepts are essential for graphics, robotics, and geographic information systems.

## Table of Contents

1. [Convex Hull Algorithms](#convex-hull-algorithms)
2. [Line Intersection](#line-intersection)
3. [Polygon Operations](#polygon-operations)
4. [Spatial Data Structures](#spatial-data-structures)
5. [Applications](#applications)
6. [Complexity Analysis](#complexity-analysis)
7. [Follow-up Questions](#follow-up-questions)

## Convex Hull Algorithms

### Theory

The convex hull of a set of points is the smallest convex polygon that contains all the points. Common algorithms include Graham scan, Jarvis march, and QuickHull.

### Convex Hull Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
    "sort"
)

type Point struct {
    X, Y float64
}

type ConvexHull struct {
    Points []Point
}

func NewConvexHull(points []Point) *ConvexHull {
    return &ConvexHull{
        Points: points,
    }
}

func (ch *ConvexHull) GrahamScan() []Point {
    if len(ch.Points) < 3 {
        return ch.Points
    }
    
    // Find the bottom-most point (and leftmost in case of tie)
    start := ch.findBottomLeft()
    
    // Sort points by polar angle with respect to start point
    sortedPoints := ch.sortByPolarAngle(start)
    
    // Build convex hull
    hull := []Point{sortedPoints[0], sortedPoints[1]}
    
    for i := 2; i < len(sortedPoints); i++ {
        // Remove points that make a non-left turn
        for len(hull) > 1 && ch.orientation(hull[len(hull)-2], hull[len(hull)-1], sortedPoints[i]) != 2 {
            hull = hull[:len(hull)-1]
        }
        hull = append(hull, sortedPoints[i])
    }
    
    return hull
}

func (ch *ConvexHull) JarvisMarch() []Point {
    if len(ch.Points) < 3 {
        return ch.Points
    }
    
    // Find the leftmost point
    leftmost := 0
    for i := 1; i < len(ch.Points); i++ {
        if ch.Points[i].X < ch.Points[leftmost].X {
            leftmost = i
        }
    }
    
    var hull []Point
    p := leftmost
    q := 0
    
    for {
        hull = append(hull, ch.Points[p])
        q = (p + 1) % len(ch.Points)
        
        for i := 0; i < len(ch.Points); i++ {
            if ch.orientation(ch.Points[p], ch.Points[i], ch.Points[q]) == 2 {
                q = i
            }
        }
        
        p = q
        
        if p == leftmost {
            break
        }
    }
    
    return hull
}

func (ch *ConvexHull) QuickHull() []Point {
    if len(ch.Points) < 3 {
        return ch.Points
    }
    
    // Find leftmost and rightmost points
    leftmost := 0
    rightmost := 0
    
    for i := 1; i < len(ch.Points); i++ {
        if ch.Points[i].X < ch.Points[leftmost].X {
            leftmost = i
        }
        if ch.Points[i].X > ch.Points[rightmost].X {
            rightmost = i
        }
    }
    
    var hull []Point
    
    // Recursively find points on convex hull
    hull = append(hull, ch.quickHullRecursive(ch.Points, ch.Points[leftmost], ch.Points[rightmost], 1)...)
    hull = append(hull, ch.quickHullRecursive(ch.Points, ch.Points[rightmost], ch.Points[leftmost], 1)...)
    
    return hull
}

func (ch *ConvexHull) quickHullRecursive(points []Point, p1, p2 Point, side int) []Point {
    var hull []Point
    maxDist := 0.0
    maxIndex := -1
    
    // Find the point with maximum distance from the line
    for i, point := range points {
        dist := ch.distanceFromLine(p1, p2, point)
        if ch.orientation(p1, p2, point) == side && dist > maxDist {
            maxDist = dist
            maxIndex = i
        }
    }
    
    if maxIndex == -1 {
        return []Point{p1, p2}
    }
    
    // Recursively find points on both sides
    hull = append(hull, ch.quickHullRecursive(points, points[maxIndex], p1, -ch.orientation(points[maxIndex], p1, p2))...)
    hull = append(hull, ch.quickHullRecursive(points, points[maxIndex], p2, -ch.orientation(points[maxIndex], p2, p1))...)
    
    return hull
}

func (ch *ConvexHull) findBottomLeft() int {
    bottomLeft := 0
    for i := 1; i < len(ch.Points); i++ {
        if ch.Points[i].Y < ch.Points[bottomLeft].Y || 
           (ch.Points[i].Y == ch.Points[bottomLeft].Y && ch.Points[i].X < ch.Points[bottomLeft].X) {
            bottomLeft = i
        }
    }
    return bottomLeft
}

func (ch *ConvexHull) sortByPolarAngle(start int) []Point {
    startPoint := ch.Points[start]
    points := make([]Point, len(ch.Points))
    copy(points, ch.Points)
    
    // Move start point to beginning
    points[0], points[start] = points[start], points[0]
    
    // Sort by polar angle
    sort.Slice(points[1:], func(i, j int) bool {
        i++ // Adjust for slice offset
        j++
        o := ch.orientation(startPoint, points[i], points[j])
        if o == 0 {
            return ch.distance(startPoint, points[i]) < ch.distance(startPoint, points[j])
        }
        return o == 2
    })
    
    return points
}

func (ch *ConvexHull) orientation(p, q, r Point) int {
    val := (q.Y-p.Y)*(r.X-q.X) - (q.X-p.X)*(r.Y-q.Y)
    if val == 0 {
        return 0 // Collinear
    }
    if val > 0 {
        return 1 // Clockwise
    }
    return 2 // Counterclockwise
}

func (ch *ConvexHull) distance(p1, p2 Point) float64 {
    dx := p2.X - p1.X
    dy := p2.Y - p1.Y
    return math.Sqrt(dx*dx + dy*dy)
}

func (ch *ConvexHull) distanceFromLine(p1, p2, p3 Point) float64 {
    return math.Abs((p2.Y-p1.Y)*p3.X - (p2.X-p1.X)*p3.Y + p2.X*p1.Y - p2.Y*p1.X) / ch.distance(p1, p2)
}

func (ch *ConvexHull) CalculateArea() float64 {
    hull := ch.GrahamScan()
    if len(hull) < 3 {
        return 0
    }
    
    area := 0.0
    n := len(hull)
    
    for i := 0; i < n; i++ {
        j := (i + 1) % n
        area += hull[i].X * hull[j].Y
        area -= hull[j].X * hull[i].Y
    }
    
    return math.Abs(area) / 2.0
}

func (ch *ConvexHull) CalculatePerimeter() float64 {
    hull := ch.GrahamScan()
    if len(hull) < 2 {
        return 0
    }
    
    perimeter := 0.0
    n := len(hull)
    
    for i := 0; i < n; i++ {
        j := (i + 1) % n
        perimeter += ch.distance(hull[i], hull[j])
    }
    
    return perimeter
}

func main() {
    points := []Point{
        {0, 3}, {1, 1}, {2, 2}, {4, 4}, {0, 0}, {1, 2}, {3, 1}, {3, 3},
    }
    
    ch := NewConvexHull(points)
    
    fmt.Println("Computational Geometry Demo:")
    fmt.Printf("Input points: %v\n", points)
    
    // Graham Scan
    hull := ch.GrahamScan()
    fmt.Printf("Convex Hull (Graham Scan): %v\n", hull)
    
    // Jarvis March
    hull2 := ch.JarvisMarch()
    fmt.Printf("Convex Hull (Jarvis March): %v\n", hull2)
    
    // QuickHull
    hull3 := ch.QuickHull()
    fmt.Printf("Convex Hull (QuickHull): %v\n", hull3)
    
    // Calculate area and perimeter
    area := ch.CalculateArea()
    perimeter := ch.CalculatePerimeter()
    fmt.Printf("Area: %.2f\n", area)
    fmt.Printf("Perimeter: %.2f\n", perimeter)
}
```

## Line Intersection

### Theory

Line intersection algorithms determine if and where two line segments intersect. This is fundamental for collision detection, computer graphics, and geometric computations.

### Line Intersection Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

type LineSegment struct {
    Start, End Point
}

type LineIntersection struct {
    Intersects bool
    Point      Point
    Type       string // "point", "segment", "none"
}

func NewLineSegment(start, end Point) *LineSegment {
    return &LineSegment{
        Start: start,
        End:   end,
    }
}

func (ls *LineSegment) IntersectsWith(other *LineSegment) LineIntersection {
    // Check if line segments intersect
    o1 := orientation(ls.Start, ls.End, other.Start)
    o2 := orientation(ls.Start, ls.End, other.End)
    o3 := orientation(other.Start, other.End, ls.Start)
    o4 := orientation(other.Start, other.End, ls.End)
    
    // General case
    if o1 != o2 && o3 != o4 {
        intersection := ls.findIntersectionPoint(other)
        return LineIntersection{
            Intersects: true,
            Point:      intersection,
            Type:       "point",
        }
    }
    
    // Special cases
    if o1 == 0 && ls.onSegment(other.Start) {
        return LineIntersection{
            Intersects: true,
            Point:      other.Start,
            Type:       "point",
        }
    }
    
    if o2 == 0 && ls.onSegment(other.End) {
        return LineIntersection{
            Intersects: true,
            Point:      other.End,
            Type:       "point",
        }
    }
    
    if o3 == 0 && other.onSegment(ls.Start) {
        return LineIntersection{
            Intersects: true,
            Point:      ls.Start,
            Type:       "point",
        }
    }
    
    if o4 == 0 && other.onSegment(ls.End) {
        return LineIntersection{
            Intersects: true,
            Point:      ls.End,
            Type:       "point",
        }
    }
    
    return LineIntersection{
        Intersects: false,
        Type:       "none",
    }
}

func (ls *LineSegment) onSegment(p Point) bool {
    return p.X <= math.Max(ls.Start.X, ls.End.X) &&
           p.X >= math.Min(ls.Start.X, ls.End.X) &&
           p.Y <= math.Max(ls.Start.Y, ls.End.Y) &&
           p.Y >= math.Min(ls.Start.Y, ls.End.Y)
}

func (ls *LineSegment) findIntersectionPoint(other *LineSegment) Point {
    // Calculate intersection point using parametric equations
    x1, y1 := ls.Start.X, ls.Start.Y
    x2, y2 := ls.End.X, ls.End.Y
    x3, y3 := other.Start.X, other.Start.Y
    x4, y4 := other.End.X, other.End.Y
    
    denom := (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    
    if math.Abs(denom) < 1e-10 {
        // Lines are parallel
        return Point{0, 0}
    }
    
    t := ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    
    x := x1 + t*(x2-x1)
    y := y1 + t*(y2-y1)
    
    return Point{x, y}
}

func (ls *LineSegment) DistanceToPoint(p Point) float64 {
    // Calculate distance from point to line segment
    A := p.X - ls.Start.X
    B := p.Y - ls.Start.Y
    C := ls.End.X - ls.Start.X
    D := ls.End.Y - ls.Start.Y
    
    dot := A*C + B*D
    lenSq := C*C + D*D
    
    if lenSq == 0 {
        // Line segment is actually a point
        return math.Sqrt(A*A + B*B)
    }
    
    param := dot / lenSq
    
    var xx, yy float64
    
    if param < 0 {
        xx, yy = ls.Start.X, ls.Start.Y
    } else if param > 1 {
        xx, yy = ls.End.X, ls.End.Y
    } else {
        xx = ls.Start.X + param*C
        yy = ls.Start.Y + param*D
    }
    
    dx := p.X - xx
    dy := p.Y - yy
    
    return math.Sqrt(dx*dx + dy*dy)
}

func (ls *LineSegment) Length() float64 {
    dx := ls.End.X - ls.Start.X
    dy := ls.End.Y - ls.Start.Y
    return math.Sqrt(dx*dx + dy*dy)
}

func (ls *LineSegment) Slope() float64 {
    dx := ls.End.X - ls.Start.X
    dy := ls.End.Y - ls.Start.Y
    
    if dx == 0 {
        return math.Inf(1) // Vertical line
    }
    
    return dy / dx
}

func orientation(p, q, r Point) int {
    val := (q.Y-p.Y)*(r.X-q.X) - (q.X-p.X)*(r.Y-q.Y)
    if val == 0 {
        return 0 // Collinear
    }
    if val > 0 {
        return 1 // Clockwise
    }
    return 2 // Counterclockwise
}

func main() {
    // Test line intersection
    ls1 := NewLineSegment(Point{0, 0}, Point{4, 4})
    ls2 := NewLineSegment(Point{0, 4}, Point{4, 0})
    
    intersection := ls1.IntersectsWith(ls2)
    fmt.Printf("Line segments intersect: %v\n", intersection.Intersects)
    if intersection.Intersects {
        fmt.Printf("Intersection point: (%.2f, %.2f)\n", intersection.Point.X, intersection.Point.Y)
    }
    
    // Test distance to point
    point := Point{2, 2}
    distance := ls1.DistanceToPoint(point)
    fmt.Printf("Distance from point (2,2) to line segment: %.2f\n", distance)
    
    // Test line properties
    fmt.Printf("Line length: %.2f\n", ls1.Length())
    fmt.Printf("Line slope: %.2f\n", ls1.Slope())
}
```

## Polygon Operations

### Theory

Polygon operations include point-in-polygon testing, polygon intersection, area calculation, and convexity checking. These are essential for geometric algorithms and spatial queries.

### Polygon Operations Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

type Polygon struct {
    Vertices []Point
}

func NewPolygon(vertices []Point) *Polygon {
    return &Polygon{
        Vertices: vertices,
    }
}

func (p *Polygon) IsPointInside(point Point) bool {
    if len(p.Vertices) < 3 {
        return false
    }
    
    inside := false
    n := len(p.Vertices)
    
    for i, j := 0, n-1; i < n; j, i = i, i+1 {
        if ((p.Vertices[i].Y > point.Y) != (p.Vertices[j].Y > point.Y)) &&
           (point.X < (p.Vertices[j].X-p.Vertices[i].X)*(point.Y-p.Vertices[i].Y)/(p.Vertices[j].Y-p.Vertices[i].Y)+p.Vertices[i].X) {
            inside = !inside
        }
    }
    
    return inside
}

func (p *Polygon) CalculateArea() float64 {
    if len(p.Vertices) < 3 {
        return 0
    }
    
    area := 0.0
    n := len(p.Vertices)
    
    for i := 0; i < n; i++ {
        j := (i + 1) % n
        area += p.Vertices[i].X * p.Vertices[j].Y
        area -= p.Vertices[j].X * p.Vertices[i].Y
    }
    
    return math.Abs(area) / 2.0
}

func (p *Polygon) CalculatePerimeter() float64 {
    if len(p.Vertices) < 2 {
        return 0
    }
    
    perimeter := 0.0
    n := len(p.Vertices)
    
    for i := 0; i < n; i++ {
        j := (i + 1) % n
        dx := p.Vertices[j].X - p.Vertices[i].X
        dy := p.Vertices[j].Y - p.Vertices[i].Y
        perimeter += math.Sqrt(dx*dx + dy*dy)
    }
    
    return perimeter
}

func (p *Polygon) IsConvex() bool {
    if len(p.Vertices) < 3 {
        return false
    }
    
    n := len(p.Vertices)
    prev := 0
    curr := 0
    
    for i := 0; i < n; i++ {
        curr = orientation(p.Vertices[i], p.Vertices[(i+1)%n], p.Vertices[(i+2)%n])
        
        if curr != 0 {
            if curr != prev && prev != 0 {
                return false
            }
            prev = curr
        }
    }
    
    return true
}

func (p *Polygon) GetCentroid() Point {
    if len(p.Vertices) == 0 {
        return Point{0, 0}
    }
    
    cx, cy := 0.0, 0.0
    n := len(p.Vertices)
    
    for _, vertex := range p.Vertices {
        cx += vertex.X
        cy += vertex.Y
    }
    
    return Point{cx / float64(n), cy / float64(n)}
}

func (p *Polygon) GetBoundingBox() (Point, Point) {
    if len(p.Vertices) == 0 {
        return Point{0, 0}, Point{0, 0}
    }
    
    minX, minY := p.Vertices[0].X, p.Vertices[0].Y
    maxX, maxY := p.Vertices[0].X, p.Vertices[0].Y
    
    for _, vertex := range p.Vertices[1:] {
        if vertex.X < minX {
            minX = vertex.X
        }
        if vertex.X > maxX {
            maxX = vertex.X
        }
        if vertex.Y < minY {
            minY = vertex.Y
        }
        if vertex.Y > maxY {
            maxY = vertex.Y
        }
    }
    
    return Point{minX, minY}, Point{maxX, maxY}
}

func (p *Polygon) IntersectsWith(other *Polygon) bool {
    // Check if any edge of this polygon intersects with any edge of the other polygon
    for i := 0; i < len(p.Vertices); i++ {
        j := (i + 1) % len(p.Vertices)
        ls1 := NewLineSegment(p.Vertices[i], p.Vertices[j])
        
        for k := 0; k < len(other.Vertices); k++ {
            l := (k + 1) % len(other.Vertices)
            ls2 := NewLineSegment(other.Vertices[k], other.Vertices[l])
            
            if ls1.IntersectsWith(ls2).Intersects {
                return true
            }
        }
    }
    
    // Check if one polygon is completely inside the other
    if p.IsPointInside(other.Vertices[0]) || other.IsPointInside(p.Vertices[0]) {
        return true
    }
    
    return false
}

func (p *Polygon) Contains(other *Polygon) bool {
    // Check if all vertices of the other polygon are inside this polygon
    for _, vertex := range other.Vertices {
        if !p.IsPointInside(vertex) {
            return false
        }
    }
    return true
}

func (p *Polygon) Rotate(angle float64, center Point) {
    cos := math.Cos(angle)
    sin := math.Sin(angle)
    
    for i := range p.Vertices {
        // Translate to origin
        x := p.Vertices[i].X - center.X
        y := p.Vertices[i].Y - center.Y
        
        // Rotate
        newX := x*cos - y*sin
        newY := x*sin + y*cos
        
        // Translate back
        p.Vertices[i].X = newX + center.X
        p.Vertices[i].Y = newY + center.Y
    }
}

func (p *Polygon) Scale(factor float64, center Point) {
    for i := range p.Vertices {
        // Translate to origin
        x := p.Vertices[i].X - center.X
        y := p.Vertices[i].Y - center.Y
        
        // Scale
        x *= factor
        y *= factor
        
        // Translate back
        p.Vertices[i].X = x + center.X
        p.Vertices[i].Y = y + center.Y
    }
}

func main() {
    // Create a triangle
    triangle := NewPolygon([]Point{
        {0, 0}, {4, 0}, {2, 4},
    })
    
    fmt.Println("Polygon Operations Demo:")
    
    // Test point inside
    point := Point{2, 1}
    inside := triangle.IsPointInside(point)
    fmt.Printf("Point (2,1) is inside triangle: %v\n", inside)
    
    // Calculate area and perimeter
    area := triangle.CalculateArea()
    perimeter := triangle.CalculatePerimeter()
    fmt.Printf("Triangle area: %.2f\n", area)
    fmt.Printf("Triangle perimeter: %.2f\n", perimeter)
    
    // Check if convex
    convex := triangle.IsConvex()
    fmt.Printf("Triangle is convex: %v\n", convex)
    
    // Get centroid
    centroid := triangle.GetCentroid()
    fmt.Printf("Triangle centroid: (%.2f, %.2f)\n", centroid.X, centroid.Y)
    
    // Get bounding box
    min, max := triangle.GetBoundingBox()
    fmt.Printf("Bounding box: (%.2f, %.2f) to (%.2f, %.2f)\n", min.X, minY, max.X, max.Y)
    
    // Create another polygon
    square := NewPolygon([]Point{
        {1, 1}, {3, 1}, {3, 3}, {1, 3},
    })
    
    // Test intersection
    intersects := triangle.IntersectsWith(square)
    fmt.Printf("Triangle intersects with square: %v\n", intersects)
    
    // Test containment
    contains := triangle.Contains(square)
    fmt.Printf("Triangle contains square: %v\n", contains)
}
```

## Follow-up Questions

### 1. Convex Hull Algorithms
**Q: What are the time complexities of different convex hull algorithms?**
A: Graham scan: O(n log n), Jarvis march: O(nh) where h is the number of hull points, QuickHull: O(n log n) average case, O(n²) worst case.

### 2. Line Intersection
**Q: How do you handle numerical precision issues in line intersection calculations?**
A: Use epsilon values for floating-point comparisons, avoid division by very small numbers, and consider using exact arithmetic for critical applications.

### 3. Polygon Operations
**Q: What is the ray casting algorithm for point-in-polygon testing?**
A: Cast a ray from the point to infinity and count intersections with polygon edges. Odd number of intersections means the point is inside.

## Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Graham Scan | O(n log n) | O(n) | Most commonly used |
| Jarvis March | O(nh) | O(h) | Good for small hulls |
| QuickHull | O(n log n) | O(n) | Average case |
| Line Intersection | O(1) | O(1) | Constant time |
| Point in Polygon | O(n) | O(1) | Linear in vertices |

## Applications

1. **Convex Hull**: Computer graphics, collision detection, pattern recognition
2. **Line Intersection**: CAD systems, game engines, geographic information systems
3. **Polygon Operations**: Spatial databases, computational geometry, computer vision
4. **Computational Geometry**: Robotics, computer graphics, geographic information systems

---

**Next**: [Number Theory Algorithms](number-theory-algorithms.md) | **Previous**: [Advanced Algorithms](README.md) | **Up**: [Advanced Algorithms](README.md)
