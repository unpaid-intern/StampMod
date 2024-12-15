extends Spatial

onready var ray = $RayCast
var ground_y = null
var collided_object = null

func detect_collision_at(position: Vector3):
	# Move the RayDetector to the specified position
	global_transform.origin = position

	# Reset collision results
	ground_y = null
	collided_object = null

	# Check if the RayCast is colliding
	if ray.is_colliding():
		var collision_point = ray.get_collision_point()
		ground_y = collision_point.y

func get_ground_y():
	return ground_y
