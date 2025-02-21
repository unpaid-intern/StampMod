#HUGE thanks to Eli who basically wrote this entire thing

extends Node

onready var canvas_handler = get_tree().get_nodes_in_group("canvasspawner")[0]


var moveSpeed = 0
var fastSpeed = 10
var rotateSpeed = 100
var fastRotateSpeed = 100
var offset = Vector3(0, -1, 0)

var inputValues = {
	"I": 0, "K": 0,
	"J": 0, "L": 0,
	"U": 0, "O": 0,
	"up": 0, "down": 0,
	"left": 0, "right": 0
}

#2canvas stuff
var origininitial = Vector3(0, 0, 0)     # initial center point
var screen1initial = Vector3(-1, 0, 0)     # e.g. initial offset of screen1 relative to center
var screen2initial = Vector3(1, 0, 0)      # e.g. initial offset of screen2 relative to center
var rotinitial = Vector3(0, 0, 0)          # initial rotation (Euler angles, in radians)
var current_rotation = rotinitial          # will be updated with yaw/pitch input

#
#lazy
#
var defaultgif

var framedatapath
var temp = 0 #delete me
var sparecanvasid
var sparecanvasexists
var lp
var ray_detector
var cam
var _hud
var _actor_man
var current_zone = "main_zone"
var last_mouse_pos
var ondefault
# current stamp info
var canvastotal = 0
var imgx
var imgy
var groundoffset
var large #using 2 canvases
var vertical # if canvases are alligned vertical if large
var isgif
var stamppath
var framepath
# variables to keep track of center position (there will be a center regardless of using 1 or 2 canvases)
var base_y = null
var centerpos
var centerspacing #only relevant if large is true
var centerrot #rotation, important for large canvas management
var stampdir #facing position
# stamp gif info
enum PlaybackMode{NORMAL, HALF, SLOW, MANUAL}
var playback_mode = PlaybackMode.NORMAL
var _playing
var _framecount
var _framedelay
var manual_frame_index = 0
var _current_frame_index = 0
var frame_data = []
var frame_delays = []
# The "screen" canvas references
var screen1ActorID
var screen1Actor
var screen1CanvasNode
var screen1TileMap

# The "controller" canvas references
var screen2ActorID
var screen2Actor
var screen2CanvasNode
var screen2TileMap

onready var ray_detector_scene = preload("res://mods/PurplePuppy-EaselAPI/RayDetector.tscn")
onready var _PlayerData = get_node_or_null("/root/PlayerData")
onready var _OptionsMenu = get_node_or_null("/root/OptionsMenu")
onready var _Network = get_node_or_null("/root/Network")

var moveVector = Vector3.ZERO
var oldMoveSpeed = moveSpeed
var oldRotateSpeed = rotateSpeed

func _ready():
	update_dynamic_nodes()
	canvas_handler.connect("spawncanvas", self, "spawnCanvas")
	canvas_handler.connect("togglemode", self, "togglemode")
	canvas_handler.connect("playgif", self, "playgif")
	canvas_handler.connect("resetgif", self, "resetgif")
	canvas_handler.connect("ctrlz", self, "ctrlz")
	_PlayerData.connect("_chalk_update", self, "_chalk_update")
	ray_detector = ray_detector_scene.instance()
	get_tree().current_scene.add_child(ray_detector)
	ray_detector.translation = Vector3.ZERO
	SceneTransition.connect("_finished", self, "pausephysics")

func pausephysics():
	pausephysics = true
	var wasplaying = _playing
	_playing = false
	yield(get_tree().create_timer(2), "timeout")
	if screen1exists:
		pausephysics = false
	_playing = wasplaying

func update_dynamic_nodes():
	if not cam:
		var viewport = get_node_or_null("/root/world/Viewport")
		if viewport:
			cam = viewport.get_camera()
		else:
			return false
	if not _hud:
		_hud = get_node_or_null("/root/playerhud")
		if not _hud:
			return false
	if not lp:
		lp = get_tree().current_scene.get_node_or_null("Viewport/main/entities/player")
		if not lp:
			return false
	if not _actor_man:
		_actor_man = get_node_or_null("/root/world")
		if not _actor_man:
			return false
	if not ray_detector: #nonexistant function get parent on base nil what am i doing wring
		ray_detector = ray_detector_scene.instance()
		get_tree().current_scene.add_child(ray_detector)
		ray_detector.translation = Vector3.ZERO
	if PlayerData.player_saved_zone:
		current_zone = PlayerData.player_saved_zone
	return true

var pausephysics = true

func _physics_process(delta):
	if not cam:
		return
	if not screen1Actor or pausephysics:
		return
	if large and canvastotal == 2 and not screen2Actor:
		return

	var cam_yaw = cam.global_transform.basis.get_euler().y
	var forward = Vector3(0, 0, -1).rotated(Vector3.UP, cam_yaw)
	var right   = Vector3(1, 0, 0).rotated(Vector3.UP, cam_yaw)


	var forward_input  = inputValues["I"] - inputValues["K"]
	var right_input    = inputValues["L"] - inputValues["J"]
	var vertical_input = inputValues["O"] - inputValues["U"]
	var move_hor = (forward * forward_input + right * right_input) * (delta * moveSpeed)


	if screen1exists and !(large and canvastotal == 2):
		# Horizontal movement
		if move_hor != Vector3.ZERO:
			var pos = screen1Actor.global_transform.origin
			pos.x += move_hor.x
			pos.z += move_hor.z
			screen1Actor.global_transform.origin = pos
			centerpos = screen1Actor.global_transform.origin

		# Rotation input
		var yaw   = (inputValues["right"] - inputValues["left"]) * delta * rotateSpeed
		var pitch = (inputValues["down"] - inputValues["up"])     * delta * rotateSpeed

		if yaw != 0:
			screen1Actor.rotate(Vector3.UP, deg2rad(yaw))
		if pitch != 0:
			screen1Actor.rotate_object_local(Vector3.RIGHT, deg2rad(pitch))

		# Clamp pitch so it never goes below 0.
		if screen1Actor.rotation.x < 0:
			var fixed = screen1Actor.rotation
			fixed.x = 0
			screen1Actor.rotation = fixed

		# Initialize base_y on first run.
		if base_y == null:
			base_y = _gety(screen1Actor.global_transform.origin) + groundoffset

		# Compute vertical lift based on pitch.
		var current_pitch = clamp(screen1Actor.rotation.x, 0, PI)
		var new_y = base_y + groundoffset * sin(current_pitch)
		var new_transform = screen1Actor.global_transform
		new_transform.origin.y = new_y
		screen1Actor.global_transform = new_transform

		# Allow vertical input to adjust base_y.
		base_y += vertical_input * delta * moveSpeed

	# ---- TWO CANVAS LOGIC ----
	else:
		# 1. Update the center position (like the single-canvas horizontal movement)
		if move_hor != Vector3.ZERO:
			centerpos.x += move_hor.x
			centerpos.z += move_hor.z

		# 2. Update the current rotation from yaw/pitch input.
		var yaw_delta   = (inputValues["right"] - inputValues["left"]) * delta * rotateSpeed
		var pitch_delta = (inputValues["down"] - inputValues["up"])     * delta * rotateSpeed
		current_rotation.y += deg2rad(yaw_delta)
		current_rotation.x += deg2rad(pitch_delta)
		# Clamp pitch between 0 and PI (0° to 180°)
		current_rotation.x = clamp(current_rotation.x, 0, PI)

		# 3. Update vertical position of the center using the same logic.
		if base_y == null:
			base_y = _gety(centerpos) + groundoffset
		base_y += vertical_input * delta * moveSpeed
		var new_y = base_y + groundoffset * sin(current_rotation.x)
		centerpos.y = new_y

		# 4. Compute each canvas’s offset relative to the center.
		var offset1 = screen1initial - origininitial
		var offset2 = screen2initial - origininitial

		# 5. Calculate the "delta rotation" – how much the current rotation differs from the initial.
		var b_current = Basis(current_rotation)
		var b_initial = Basis(rotinitial)
		var delta_rot = b_current * b_initial.inverse()

		# 6. Rotate the initial offsets.
		var rotated_offset1 = delta_rot.xform(offset1)
		var rotated_offset2 = delta_rot.xform(offset2)

		# 7. Set the new positions for each canvas around the center.
		screen1Actor.global_transform.origin = centerpos + rotated_offset1
		screen2Actor.global_transform.origin = centerpos + rotated_offset2

		# 8. Update both canvases' rotations to match current_rotation.
		screen1Actor.rotation = current_rotation
		screen2Actor.rotation = current_rotation


	var inputChanged = false
	if move_hor != Vector3.ZERO:
		inputChanged = true
	elif (inputValues["I"] - inputValues["K"]) != 0:
		inputChanged = true
	elif (inputValues["L"] - inputValues["J"]) != 0:
		inputChanged = true
	elif (inputValues["O"] - inputValues["U"]) != 0:
		inputChanged = true
	elif (inputValues["right"] - inputValues["left"]) != 0:
		inputChanged = true
	elif (inputValues["down"] - inputValues["up"]) != 0:
		inputChanged = true

	if inputChanged:
		posUpdate()




func _input(event):
	if not event is InputEventKey:
		return

	if event.pressed:
		if lp and not lp.busy:
			match event.scancode:
				KEY_I:   inputValues["I"] = 1
				KEY_K:   inputValues["K"] = 1
				KEY_J:   inputValues["J"] = 1
				KEY_L:   inputValues["L"] = 1
				KEY_U:   inputValues["U"] = 0.5
				KEY_O:   inputValues["O"] = 0.5
				KEY_UP:    inputValues["up"] = 1
				KEY_DOWN:  inputValues["down"] = 1
				KEY_LEFT:  inputValues["left"] = 1
				KEY_RIGHT: inputValues["right"] = 1
				KEY_CONTROL:
					moveSpeed   = fastSpeed
					rotateSpeed = fastRotateSpeed
	else:
		match event.scancode:
			KEY_I:   inputValues["I"] = 0
			KEY_K:   inputValues["K"] = 0
			KEY_J:   inputValues["J"] = 0
			KEY_L:   inputValues["L"] = 0
			KEY_U:   inputValues["U"] = 0
			KEY_O:   inputValues["O"] = 0
			KEY_UP:    inputValues["up"] = 0
			KEY_DOWN:  inputValues["down"] = 0
			KEY_LEFT:  inputValues["left"] = 0
			KEY_RIGHT: inputValues["right"] = 0
			KEY_CONTROL:
				moveSpeed   = oldMoveSpeed
				rotateSpeed = oldRotateSpeed

#
# CANVAS CREATION & DRAWING
#
func parse_stamp_data(path):
	var file = File.new()
	var error = file.open(path, File.READ)
	if error != OK:
		push_error("Could not open file: " + path)
		return {}

	# Default values (in case they aren't specified in the file)
	imgx = 0.0
	imgy = 0.0
	isgif = false
	resetGifState()
	_framecount = 0
	_framedelay = false

	if not file.eof_reached():
		var size_line = file.get_line().strip_edges()
		var size_parts = size_line.split(",")

		# We expect at least 3 parts: (imgx, imgy, type)
		if size_parts.size() >= 3:
			imgx = size_parts[0].to_float()
			imgy = size_parts[1].to_float()

			if size_parts[2] == "gif":
				isgif = true
				# For gif, expect at least 5 parts: (imgx, imgy, type, framecount, framedelay)
				if size_parts.size() >= 5:
					_framecount = size_parts[3].to_int()

					var candidate_delay = size_parts[4].to_int()
					if candidate_delay == -1 or candidate_delay == 0:
						_framedelay = false
					else:
						_framedelay = candidate_delay

	file.close()

	# Return the extracted values in a Dictionary
	return {
		"imgx": imgx,
		"imgy": imgy,
		"isgif": isgif,
		"framecount": _framecount,
		"framedelay": _framedelay
	}

func assign_stamp_data(data: Dictionary) -> void:
	if data.size() == 0:
		print("No data to apply or failed to parse.")
		return

	# Example of assigning to local/script variables
	groundoffset = (data["imgy"]) / 2
	var worldyunits = (data["imgx"]) / 2
	imgx = int(round(float(data["imgx"]) * 10))
	imgy = int(round(float(data["imgy"]) * 10))
	isgif = data["isgif"]
	_framecount = data["framecount"]
	_framedelay = data["framedelay"]
	if imgx > 200 || imgy > 200:
		large = true
		if imgy > imgx:
			vertical = true
			centerspacing = data["imgy"] - 20.0
		else:
			vertical = false
			centerspacing = data["imgx"] - 20.0
		print(centerspacing)
	else:
		large = false


func getstampdata(path: String) -> Array:
	var result_data = []

	var file = File.new()
	var error = file.open(path, File.READ)
	if error != OK:
		push_error("Could not open file: " + path)
		return result_data

	# Skip the first line (the one containing imgx, imgy, gif info, etc.)
	if not file.eof_reached():
		file.get_line() # discard first line

	# Now read subsequent lines for tile data
	while not file.eof_reached():
		var line = file.get_line().strip_edges()
		if line != "":
			var parts = line.split(",")
			if parts.size() == 3:
				# Multiply parts 0 and 1 by 10, round them, then convert to int
				var img_x = int(round(parts[0].to_float() * 10))
				var img_y = int(round(parts[1].to_float() * 10))
				var color = parts[2].to_int()

				# Append an array of [x, y, colorTile] to the result_data
				result_data.append([img_x, img_y, color])

	return result_data



	file.close()
	return result_data

func applyTransform(data, imgx, imgy, firstgifframe = false):
	var transformed = []


	# Calculate centering offsets (to center the stamp in a 200×200 space)
	var offset_x = (200 - imgx) * 0.5
	var offset_y = (200 - imgy) * 0.5


	if firstgifframe:
		for tile in data:
			var orig_x = tile[0]
			var orig_y = tile[1]
			var color = tile[2]
			var a_x = orig_x + offset_x
			var a_y = orig_y + offset_y
			transformed.append([ int(round(a_x)), int(round(a_y)), color ])
	else:
		# Normal mode: flip vertically (new_y = converted_imgy - (orig_y + offset_y))
		for tile in data:
			var orig_x = tile[0]
			var orig_y = tile[1]
			var color = tile[2]
			var a_x = orig_x + offset_x
			var a_y = orig_y - offset_y
			transformed.append([ int(round(a_x)), int(round(imgy - a_y)), color ])

	return transformed

func applyLargeTransform(data, imgx, imgy):
	# For taking the input data and transforming it into two different 200×200 arrays.
	# Each array is placed on a separate canvas mat and they are placed next to each other.
	var transformed = []
	var data1 = []
	var data2 = []

	if vertical:
		var offset = imgy/2
		var offset2 = imgy-200
		var offsetx = (200 - imgx) * 0.5
		for tile in data:
			var orig_x = tile[0]
			var orig_y = tile[1]
			var color = tile[2]
			var a_x = orig_x
			var a_y = orig_y
			transformed.append([ int(round(a_x)), int(round(imgy - a_y)), color ])
		for tile in transformed:
			if tile[1] < (offset):
				data2.append([ tile[0] + offsetx, tile[1], tile[2] ])
			else:
				data1.append([ tile[0] + offsetx, tile[1] - offset2, tile[2] ])
	else:
		var offset = imgx/2
		var offset2 = imgx-200
		var offsety = (200 - imgy) * 0.5
		for tile in data:
			var orig_x = tile[0]
			var orig_y = tile[1]
			var color = tile[2]
			var a_x = orig_x
			var a_y = orig_y
			transformed.append([ int(round(a_x)), int(round(imgy - a_y)), color ])
		for tile in transformed:

			if tile[0] < (offset):
				data1.append([ tile[0], tile[1] + offsety, tile[2] ])
			else:
				data2.append([ tile[0] - offset2, tile[1] + offsety, tile[2] ])
	return {
		"load1": data1,
		"load2": data2,
	}


func rotateCanvasesToCameraYaw() -> void:
	# 1) Get the camera's current yaw (radians)
	var cam_euler: Vector3 = cam.global_transform.basis.get_euler()
	var cam_yaw: float = cam_euler.y

	# 2) Find how much we need to change the canvases' yaw
	var old_yaw: float = current_rotation.y
	var delta_yaw: float = cam_yaw - old_yaw

	# 3) Update current_rotation.y to match the camera's yaw
	current_rotation.y = cam_yaw

	# 4) Build a Basis from the yaw difference
	#    (We only rotate around Y here, so x=0, z=0 in the Euler angles)
	var delta_basis = Basis(Vector3(0, delta_yaw, 0))

	# 5) Compute each canvas's initial offset from the original center
	var offset1: Vector3 = screen1initial - origininitial
	var offset2: Vector3 = screen2initial - origininitial

	# 6) Rotate these offsets by delta_basis so they pivot around centerpos
	offset1 = delta_basis.xform(offset1)
	offset2 = delta_basis.xform(offset2)

	# 7) Apply the new positions to each canvas
	screen1Actor.global_transform.origin = centerpos + offset1
	screen2Actor.global_transform.origin = centerpos + offset2

	# 8) Finally, update each canvas's rotation
	#    (We keep current_rotation.x and current_rotation.z intact, just changed y)
	screen1Actor.rotation = current_rotation
	screen2Actor.rotation = current_rotation


func getPos():
	var pos = last_mouse_pos
	pos.y = yield(_gety(last_mouse_pos), "completed")
	base_y = pos.y
	return pos

func applyInitialRotatePosition() -> void:
	# Get the camera's Euler rotation (in radians)
	var cam_euler: Vector3 = cam.global_transform.basis.get_euler()
	var cam_yaw: float = cam_euler.y
	var current_yaw: float = screen1Actor.rotation.y
	var delta_yaw: float = cam_yaw - current_yaw

	# Rotate screen1Actor (and screen2Actor if large) by the yaw difference.
	screen1Actor.rotate_object_local(Vector3.UP, delta_yaw)
	rotinitial = screen1Actor.global_transform.basis.get_euler()
	if large:
		screen2Actor.rotate_object_local(Vector3.UP, delta_yaw)



func getLargePos(centerpos: Vector3, centerspacing: float) -> Dictionary:
	var pos1: Vector3
	var pos2: Vector3
	if vertical:
		# Compute a horizontal forward vector based only on the camera's yaw.
		var cam_yaw: float = cam.global_transform.basis.get_euler().y
		var forward: Vector3 = Vector3(0, 0, -1).rotated(Vector3.UP, cam_yaw)
		pos1 = centerpos - forward * (centerspacing * 0.5)
		pos2 = centerpos + forward * (centerspacing * 0.5)
	else:
		# Use the camera's right vector for horizontal offset.
		var cam_right: Vector3 = cam.global_transform.basis.x.normalized()
		pos1 = centerpos - cam_right * (centerspacing * 0.5)
		pos2 = centerpos + cam_right * (centerspacing * 0.5)

	# Set initial positions relative to the center.
	origininitial = centerpos
	screen1initial = pos1
	screen2initial = pos2

	# Return a dictionary containing both positions.
	return {"pos1": pos1, "pos2": pos2}

var screen1exists
var waslarge

func spawnCanvas(stamppath, framespath = null):
	if not update_dynamic_nodes():
		return
	ondefault = findDefaultCanvas(last_mouse_pos)
	if (ondefault || Input.is_key_pressed(KEY_1) || Input.is_key_pressed(KEY_2) || Input.is_key_pressed(KEY_3) || Input.is_key_pressed(KEY_4)) && !(Input.is_key_pressed(KEY_CONTROL) || Input.is_key_pressed(KEY_SHIFT)):
		frames_path = framepath
		update_dynamic_nodes2()
		check_image_resolution(stamppath, last_mouse_pos)
		return
	var viewspot = false
	if Input.is_key_pressed(KEY_CONTROL):
		viewspot = true
	var _playing = false
	yield(get_tree().create_timer(0.2), "timeout")
	var atoldspot = false
	var oldspot
	var oldrot
	if centerpos && canvastotal != 0:
		oldspot = centerpos
		oldspot.y = base_y
		oldrot = screen1Actor.rotation
		if Input.is_key_pressed(KEY_SHIFT) && screen1exists:
			oldrot = screen1Actor.rotation
			atoldspot = true
	pausephysics = true
	#check if placing on canvas
	base_y = 0
	var pos = yield(getPos(), "completed")
	large = false
	current_rotation = Vector3.ZERO
	origininitial = Vector3.ZERO
	screen1initial = Vector3.ZERO
	screen2initial = Vector3.ZERO
	rotinitial = Vector3.ZERO
	centerpos = pos
	var dir = _get_player_facing_direction()
	var stampinfo = parse_stamp_data(stamppath)
	assign_stamp_data(stampinfo)
	var stampdata = getstampdata(stamppath)
	if canvastotal > 0:
		deletequery(atoldspot)
	screen1exists = true
	if large:
		var loaddata = applyLargeTransform(stampdata, imgx, imgy)
		var load1 = loaddata["load1"]
		var load2 = loaddata["load2"]
		var posdata = getLargePos(pos, centerspacing)
		var pos1 = posdata["pos1"]
		var pos2 = posdata["pos2"]

		spawnScreens(pos1, pos2, current_zone)
		waslarge = true
		canvastotal = 2
		handleScreen1Packet(load1)
		handleScreen2Packet(load2)
		applyInitialRotatePosition()
		rotateCanvasesToCameraYaw()
		if viewspot:
			var viewdata = getNearestSpot()
			var viewpos = viewdata["viewpos"]
			var viewrot = viewdata["viewrot"]
			positionCanvasesAt(viewpos, viewrot)
		elif atoldspot:
			positionCanvasesAt(oldspot, oldrot)
		pausephysics = false
	else:
		stampdata = applyTransform(stampdata, imgx, imgy, isgif)

		spawnScreen(pos, current_zone)
		canvastotal += 1
		handleScreen1Packet(stampdata)
		applyInitialRotatePosition()
		if viewspot:
			var viewdata = getNearestSpot()
			var viewpos = viewdata["viewpos"]
			var viewrot = viewdata["viewrot"]
			positionCanvasesAt(viewpos, viewrot)
		elif atoldspot:
			positionCanvasesAt(oldspot, oldrot)
		pausephysics = false
		if isgif:
			framedatapath = framespath
# Store both position and rotation for each spot. translate y -4
export(Array, Dictionary) var main_zone_spots = [
	{"viewpos": Vector3(176.26033, -0.758699, 1.404384), "viewrot": Vector3(1.192642, -1.572032, 0)},
	{"viewpos": Vector3(61.170135, 4.092033, 9.043752), "viewrot": Vector3(1.454441, 0.930119, 3.141593)},
	{"viewpos": Vector3(173.863724, -0.157966, -146.677155), "viewrot": Vector3(1.425352, 1.576408, 3.141593)},
	{"viewpos": Vector3(46.668903, 9.758698, 122.679146), "viewrot": Vector3(1.541709, -0.841171, 3.141593)}
]

export(Array, Dictionary) var non_main_zone_spots = [
	{"viewpos": Vector3(-259.851074, -18.148191, -396.667267), "viewrot": Vector3(0.959931, -1.655867, 0)}
]


func getNearestSpot() -> Dictionary:
	var candidate_spots: Array = []

	if current_zone != "main_zone":
		candidate_spots = non_main_zone_spots
	else:
		candidate_spots = main_zone_spots


	if candidate_spots.empty():
		return {"viewpos": lp.global_transform.origin, "viewrot": Vector3(0, 0, 0)}


	var nearest_candidate: Dictionary = candidate_spots[0]
	var nearest_distance: float = (lp.global_transform.origin - nearest_candidate["viewpos"]).length()

	for candidate in candidate_spots:
		var dist: float = (lp.global_transform.origin - candidate["viewpos"]).length()
		if dist < nearest_distance:
			nearest_distance = dist
			nearest_candidate = candidate

	return nearest_candidate

func positionCanvasesAt(new_center: Vector3, new_rot: Vector3) -> void:
	# 1) Extract desired pitch, yaw, and roll from new_rot.
	var new_pitch: float = new_rot.x
	var new_yaw: float = new_rot.y
	var new_roll: float = new_rot.z

	# 2) Determine the change (delta) from the current rotation.
	var old_pitch: float = current_rotation.x
	var old_yaw: float   = current_rotation.y
	var delta_pitch: float = new_pitch - old_pitch
	var delta_yaw: float   = new_yaw - old_yaw

	# 3) Update current_rotation to the new values.
	current_rotation = Vector3(new_pitch, new_yaw, new_roll)

	# 4) Update base_y if needed. If base_y is not set, compute it from the ground height.
	base_y = new_center.y
	# Optionally, update base_y here if you have any vertical adjustments.

	# 5) Adjust new_center's Y value based on the current pitch.
	var current_pitch_clamped: float = clamp(current_rotation.x, 0, PI)
	# This makes the Y position stick to a ground level plus an offset that changes with pitch.
	# 6) Set the shared center position.
	centerpos = new_center

	# 7) Single-canvas logic.
	if not (large and canvastotal == 2):
		screen1Actor.global_transform.origin = new_center
		screen1Actor.rotation = current_rotation
	else:
		# ===== Two-canvas logic: Rotate both canvases around centerpos =====
		# 7a) Compute each canvas’s original offset from the original center.
		var offset1: Vector3 = screen1initial - origininitial
		var offset2: Vector3 = screen2initial - origininitial

		# 7b) Construct a Basis for the pitch + yaw change.
		#     (We ignore roll here; include it if necessary.)
		var delta_basis = Basis(Vector3(delta_pitch, delta_yaw, 0))

		# 7c) Rotate the original offsets by delta_basis.
		offset1 = delta_basis.xform(offset1)
		offset2 = delta_basis.xform(offset2)

		# 7d) Set each canvas’s position to the center plus its rotated offset.
		screen1Actor.global_transform.origin = centerpos + offset1
		screen2Actor.global_transform.origin = centerpos + offset2

		# 7e) Update both canvases’ rotations.
		screen1Actor.rotation = current_rotation
		screen2Actor.rotation = current_rotation


func deletequery(replace):
	if large || waslarge:
		clearProps()
		canvastotal = 0
		sparecanvasid = null
	else:
		if replace:
			Network._send_actor_action(screen1ActorID, "_wipe_actor", [screen1ActorID])
			lp._wipe_actor(screen1ActorID)
			canvastotal -=1
		else:
			if canvastotal == 1:
				sparecanvasid = screen1ActorID
				sparecanvasexists = true
			elif canvastotal == 2:
				Network._send_actor_action(sparecanvasid, "_wipe_actor", [sparecanvasid])
				lp._wipe_actor(sparecanvasid)
				sparecanvasid = screen1ActorID
				canvastotal -=1
			else:
				clearProps()
				canvastotal = 0
				sparecanvasid = null
#sensors
func _gety(pos):
	if ray_detector:
		var ray_position = pos + Vector3(0, 20, 0)
		ray_detector.detect_collision_at(ray_position)
		yield (get_tree().create_timer(1.0 / 80.0), "timeout")
		ray_detector.detect_collision_at(ray_position)
		yield (get_tree().create_timer(1.0 / 80.0), "timeout")
		ray_detector.detect_collision_at(ray_position)
		var ground_y = ray_detector.get_ground_y()
		if ground_y != null:
			ground_y -=  0.0086999999999999
			return ground_y
		else:
			pos.y -= 0.9868
			return pos.y

func _get_player_facing_direction():
	if cam:
		var forward = cam.global_transform.basis.z.normalized()
		if abs(forward.x) > abs(forward.z):
			return "right" if forward.x > 0 else "left"
		else:
			return "down" if forward.z > 0 else "up"

func _chalk_update(pos):
	last_mouse_pos = pos
#
#drawing on in game canvases
#


func findDefaultCanvas(pos: Vector3):
	if current_zone != "main_zone":
		return false
	# Grid 1 check
	if (pos.x > 48.571999 - 10 and pos.x < 48.571999 + 10) and (pos.z > -51.041 - 10 and pos.z < -51.041 + 10):
		return true
	# Grid 2 check
	elif (pos.x > 69.57199900000001 - 10 and pos.x < 69.57199900000001 + 10) and (pos.z > -54.952999 - 10 and pos.z < -54.952999 + 10):
		return true
	# Grid 3 check
	elif (pos.x > -54.7896 - 10 and pos.x < -54.7896 + 10) and (pos.z > -115.719002 - 10 and pos.z < -115.719002 + 10):
		return true
	# Grid 4 check
	elif (pos.x > -25.781099 - 10 and pos.x < -25.781099 + 10) and (pos.z > -34.5681 - 10 and pos.z < -34.5681 + 10):
		return true
	else:
		# Position not in any grid range
		return false

#canvas creation

func createCanvas(targetPos, zone):
	var canvasResult = {}
	canvasResult["actorID"] = Network._sync_create_actor("canvas", targetPos, zone, -1, Network.STEAM_ID, Vector3.ZERO)

	for node in get_tree().get_nodes_in_group("actor"):
		if not is_instance_valid(node):
			continue
		if node.actor_id == canvasResult["actorID"]:
			canvasResult["actor"]     = node
			canvasResult["canvasNode"] = node.get_node("chalk_canvas")
			canvasResult["tileMap"]   = canvasResult["canvasNode"].get_node("Viewport/TileMap")
			break

	return canvasResult

func spawnScreen(targetPos, zone):
	print(targetPos,zone)
	var result = createCanvas(targetPos, zone)
	screen1ActorID    = result["actorID"]
	screen1Actor      = result["actor"]
	screen1CanvasNode = result["canvasNode"]
	screen1TileMap    = result["tileMap"]

func spawnScreens(targetPos, targetPos2, zone):
	var result = createCanvas(targetPos, zone)
	screen1ActorID    = result["actorID"]
	screen1Actor      = result["actor"]
	screen1CanvasNode = result["canvasNode"]
	screen1TileMap    = result["tileMap"]

	result = createCanvas(targetPos2, zone)
	screen2ActorID    = result["actorID"]
	screen2Actor      = result["actor"]
	screen2CanvasNode = result["canvasNode"]
	screen2TileMap    = result["tileMap"]

func handleScreen1Packet(data):
	# data is expected to be an array of [ [x, y, colorTile], [x, y, colorTile], ... ]
	var canvasData = []
	for pixelData in data:
		var posX = pixelData[0]
		var posY = pixelData[1]
		var colorTile = pixelData[2]
		canvasData.append([Vector2(posX, posY), colorTile])
		screen1TileMap.set_cell(posX, posY, colorTile)

	updateCanvas(canvasData, screen1ActorID)

func handleScreen2Packet(data):
	var canvasData = []
	for pixelData in data:
		var posX = pixelData[0]
		var posY = pixelData[1]
		var colorTile = pixelData[2]
		canvasData.append([Vector2(posX, posY), colorTile])
		screen2TileMap.set_cell(posX, posY, colorTile)

	updateCanvas(canvasData, screen2ActorID)

func updateCanvas(canvasData, canvasActorID):
	Network._send_P2P_Packet(
		{"type": "chalk_packet", "data": canvasData, "canvas_id": canvasActorID},
		"peers",
		2,
		Network.CHANNELS.CHALK
	)

func clearDrawings():
	var canvasData = []
	for x in range(0, 199):
		for y in range(0, 199):
			canvasData.append([Vector2(x, y), -1])
			screen1TileMap.set_cell(x, y, -1)
			screen2TileMap.set_cell(x, y, -1)

	updateCanvas(canvasData, screen1ActorID)
	updateCanvas(canvasData, screen2ActorID)

#
# MOVING OR REMOVING THE CANVAS ACTORS
#

func clearProps():
	print("clearing")
	# Removes all "canvas" actors from the world
	if screen1ActorID:
		Network._send_actor_action(screen1ActorID, "_wipe_actor", [screen1ActorID])
		lp._wipe_actor(screen1ActorID)
	if screen2ActorID:
		Network._send_actor_action(screen2ActorID, "_wipe_actor", [screen2ActorID])
		lp._wipe_actor(screen2ActorID)
	if sparecanvasid:
		Network._send_actor_action(sparecanvasid, "_wipe_actor", [sparecanvasid])
		lp._wipe_actor(sparecanvasid)
	canvastotal = 0
	screen1exists = false
	waslarge = false
	#
func posUpdate():
	# Example, only if the actor was actually moved
	Network._send_P2P_Packet({
		"type": "actor_update",
		"actor_id": screen1ActorID,
		"pos": screen1Actor.global_transform.origin,
		"rot": screen1Actor.global_rotation
	}, "peers", Network.CHANNELS.ACTOR_UPDATE)
	if large:
		Network._send_P2P_Packet({
			"type": "actor_update",
			"actor_id": screen2ActorID,
			"pos": screen2Actor.global_transform.origin,
			"rot": screen2Actor.global_rotation
		}, "peers", Network.CHANNELS.ACTOR_UPDATE)

#
# Gif Related things
#
func resetGifState() -> void:
	# Reset playback indices.
	_current_frame_index = 0
	manual_frame_index = 0
	_playing = false

	# Clear all frame data.
	if frame_data:
		frame_data.clear()


	if typeof(frame_delays) == TYPE_ARRAY:
		frame_delays.clear()


	processing = false
	another = false

	_framedelay = false

	# Send a notification for debugging (optional).
	print("GIF state has been reset.")

func applyTransformToPool(data: Array) -> PoolByteArray:
	# Pre-calculate offset values.
	var offset_x = (200 - imgx) * 0.5
	var offset_y = (200 - imgy) * 0.5
	# Preallocate: each tile gives 3 values.
	var total_values = data.size() * 3
	var transformed = PoolByteArray()
	transformed.resize(total_values)
	var idx = 0
	for tile in data:
		var orig_x = tile[0]
		var orig_y = tile[1]
		var color = tile[2]
		# Apply offsets.
		var a_x = orig_x + offset_x
		var a_y = orig_y + offset_y
		# Write directly to preallocated array.
		transformed[idx] = int(round(a_x))
		idx += 1
		transformed[idx] = int(round(a_y))
		idx += 1
		transformed[idx] = color
		idx += 1
	return transformed

func getGifData(file_path: String) -> Array:
	var final_data = []
	var file = File.new()
	if file.open(file_path, File.READ) != OK:
		push_error("Could not open file: " + file_path)
		return final_data

	# Read entire file as text.
	var file_text = file.get_as_text()
	file.close()
	
	# Split into lines.
	var lines = file_text.split("\n")
	
	var header_set = false
	var current_frame_number: int = 0
	var current_delay: int = 0
	var current_frame_data = []
	var last_transformed = null  # Store the last frame's transformed data
	
	# Loop over every line.
	for line in lines:
		line = line.strip_edges()
		if line == "":
			continue
		
		# Check if this line is a frame header.
		if line.begins_with("frame,"):
			# If we have accumulated pixel data, process it.
			if current_frame_data.size() > 0:
				var transformed = applyTransformToPool(current_frame_data)
				final_data.append([ current_frame_number, current_delay, transformed ])
				last_transformed = transformed
				current_frame_data.clear()
			else:
				# No new pixel data since the last header:
				# If we have a previous frame, duplicate it.
				if last_transformed != null:
					final_data.append([ current_frame_number, current_delay, last_transformed ])
			
			header_set = true
			var parts = line.split(",")
			if parts.size() >= 3:
				current_frame_number = parts[1].to_int()
				# Use _framedelay if it’s nonzero; otherwise, use the header delay.
				if _framedelay and int(_framedelay) != 0:
					current_delay = int(_framedelay)
				else:
					current_delay = parts[2].to_int()
				print("Parsed header: frame:", current_frame_number, "delay:", current_delay)
			else:
				push_error("Invalid frame header: " + line)
		else:
			# Process a pixel data line: expected format "x,y,color"
			var parts = line.split(",")
			if parts.size() == 3:
				var x = int(round(float(parts[0]) * 10))
				var y = int(round(float(parts[1]) * 10))
				var color = parts[2].to_int()
				current_frame_data.append([x, y, color])
			else:
				push_error("Invalid pixel data line: " + line)
	
	# Process any remaining frame data as the final frame.
	if current_frame_data.size() > 0:
		var transformed = applyTransformToPool(current_frame_data)
		final_data.append([ current_frame_number, current_delay, transformed ])
		last_transformed = transformed
	
	return final_data


func playgif(message = true):
	if not isgif:
		if message:
			PlayerData._send_notification("No gif to play!", 1)
		return

	# Load frame data if not already loaded.
	if frame_data.empty():
		frame_data = getGifData(framedatapath)
		if frame_data.empty():
			if message:
				PlayerData._send_notification("Failed to load GIF data.", 1)
			return

	# Manual mode: step a single frame per call.
	if playback_mode == PlaybackMode.MANUAL:
		if manual_frame_index >= frame_data.size():
			manual_frame_index = 0
		_play_frame(manual_frame_index)
		manual_frame_index += 1
		return

	# Automatic modes: toggle _playing on/off.
	_playing = not _playing
	if not _playing:
		if message:
			PlayerData._send_notification("Playback Stopped", 1)
		return
	else:
		if message:
			PlayerData._send_notification("Playing!", 0)
		_play()


func resetgif():
	PlayerData._send_notification("Set to frame 1", 0)
	_current_frame_index = 0
	manual_frame_index = 0
	_playing = false


func togglemode():
	var prior = playback_mode
	# Cycle through the playback modes.
	playback_mode = int(playback_mode) + 1
	if playback_mode > PlaybackMode.MANUAL:
		playback_mode = PlaybackMode.NORMAL

	var mode_names = ["Normal", "Half", "SLOW", "Manual"]
	PlayerData._send_notification("GIF mode changed to: " + mode_names[playback_mode], 0)

	# Synchronize indices when switching between Manual and automatic modes.
	if prior in [PlaybackMode.NORMAL, PlaybackMode.HALF, PlaybackMode.SLOW] and playback_mode == PlaybackMode.MANUAL:
		if _playing:
			_playing = false
		manual_frame_index = _current_frame_index

	if prior == PlaybackMode.MANUAL and playback_mode in [PlaybackMode.NORMAL, PlaybackMode.HALF, PlaybackMode.SLOW]:
		_current_frame_index = manual_frame_index

func _play():
	while _playing:
		if playback_mode == PlaybackMode.MANUAL:
			return

		_play_frame(_current_frame_index)

		var delay_ms = 0
		if _framedelay:
			delay_ms = _framedelay
		else:
			delay_ms = frame_data[_current_frame_index][1]
		var delay_sec = delay_ms / 1000.0

		match playback_mode:
			PlaybackMode.NORMAL:
				yield(get_tree().create_timer(delay_sec), "timeout")
			PlaybackMode.HALF:
				yield(get_tree().create_timer(delay_sec * 2), "timeout")
			PlaybackMode.SLOW:
				yield(get_tree().create_timer(delay_sec * 10), "timeout")

		_current_frame_index += 1
		if _current_frame_index >= frame_data.size():
			_current_frame_index = 0

		if not _playing:
			return



func _play_frame(frame_index):
	# Retrieve the pixel data PoolByteArray from the frame.
	var pool_data: PoolByteArray = frame_data[frame_index][2]
	var pixel_array = []
	# Each pixel is stored as 3 ints, so convert them.
	for i in range(0, pool_data.size(), 3):
		 var x = pool_data[i]
		 var y = pool_data[i+1]
		 var color = pool_data[i+2]
		 pixel_array.append([x, y, color])
	# Now send the converted array to handleScreen1Packet.
	handleScreen1Packet(pixel_array)



func ctrlz():
	_playing = false
	yield(get_tree().create_timer(0.2), "timeout")
	pausephysics = true
	if canvastotal != 0:
		if waslarge:
			clearProps()
			canvastotal = 0
		if canvastotal == 1 || large:
			clearProps()
			canvastotal = 0
			return
		else:
			Network._send_actor_action(sparecanvasid, "_wipe_actor", [sparecanvasid])
			lp._wipe_actor(sparecanvasid)
			sparecanvasid = screen1ActorID
			canvastotal = 1
			return
	else:
		ctrlzgamecanvas() #undo game canvas


#
# Junk to get drawing on game canvases working again.
# literally just a husk of my old puppyspawn script. BEHOLD: the first code ever written by me for non acedemic purposes. dont scream.


onready var key_handler = get_tree().get_nodes_in_group("keys")[0]
var _keybinds_api = null
var ctrlz_array = []
var _Chalknode = null
var _Smutnode = null
var _Canvas_Path = null
var _Player = null
var _Canvas = null
var _grid = []
var game_grid = null
var _tile = []
var game_tile = null
var game_canvas_id = null
var _canvas_id = []
var new_canvas = true
var _mouse_can_replace = false
var _canvas_packet
var _current_frame_index_ = 0
var send_load = []
var send_load_2 = []
var send_load_3 = []
var send_load_4 = []
var frame_data_ = []
var frame_delays_ = []

var last_mouse = null
var two = false
var dir = null
var gifdir = null
var origin = null
var _framedelay_ = null
var _framecount_ = null
var base = null
var processing = false
var img_override = false
var grid = 0
var shoulddel = false
var old_pid = 0
var another = false
enum BrushMode{
	PENCIL,
	ERASER,
}
var brush_mode = BrushMode.PENCIL
var brush_size = 1
var brush_color = 0



var img_path = null
var frames_path = null
var gui_path = null

var wait = false

var debug = true

func update_dynamic_nodes2():
	if not cam:
		var viewport = get_node_or_null("/root/world/Viewport")
		if viewport:
			cam = viewport.get_camera()
		else:
			return false
	if not _hud:
		_hud = get_node_or_null("/root/playerhud")
		if not _hud:
			return false
	if not _Player:
		_Player = get_tree().current_scene.get_node_or_null("Viewport/main/entities/player")
		if not _Player:
			return false
	if not _actor_man:
		_actor_man = get_node_or_null("/root/world")
		if not _actor_man:
			return false
	if not ray_detector.get_parent():
		get_tree().current_scene.add_child(ray_detector)
		ray_detector.translation = Vector3.ZERO
	if PlayerData.player_saved_zone:
		current_zone = PlayerData.player_saved_zone
	return true

func is_in_any_grid(pos: Vector3)->bool:
	if current_zone == "main_zone":
		if (pos.x > 48.571999 - 10 and pos.x < 48.571999 + 10) and (pos.z > - 51.041 - 10 and pos.z < - 51.041 + 10):
			return true
		elif (pos.x > 69.57199900000001 - 10 and pos.x < 69.57199900000001 + 10) and (pos.z > - 54.952999 - 10 and pos.z < - 54.952999 + 10):
			return true
		elif (pos.x > - 54.7896 - 10 and pos.x < - 54.7896 + 10) and (pos.z > - 115.719002 - 10 and pos.z < - 115.719002 + 10):
			return true
		elif (pos.x > - 25.781099 - 10 and pos.x < - 25.781099 + 10) and (pos.z > - 34.5681 - 10 and pos.z < - 34.5681 + 10):
			return true
	return false

func _spawn_canvas(pos, file_path, _offset = 10):
	if current_zone == "main_zone":
		if (pos.x > 48.571999 - 10 and pos.x < 48.571999 + 10) and (pos.z > - 51.041 - 10 and pos.z < - 51.041 + 10):
			grid = 1
		elif (pos.x > 69.57199900000001 - 10 and pos.x < 69.57199900000001 + 10) and (pos.z > - 54.952999 - 10 and pos.z < - 54.952999 + 10):
			grid = 2
		elif (pos.x > - 54.7896 - 10 and pos.x < - 54.7896 + 10) and (pos.z > - 115.719002 - 10 and pos.z < - 115.719002 + 10):
			grid = 3
		elif (pos.x > - 25.781099 - 10 and pos.x < - 25.781099 + 10) and (pos.z > - 34.5681 - 10 and pos.z < - 34.5681 + 10):
			grid = 4
		else:
			grid = 0
	else:
		grid = 0

	if grid == 0 && key_handler.stupidincompatabilitydontpatchshitthatbreaksotherpeoplesmods:
			PlayerData._send_notification("feature incompatable with \"thetamborine\" mod", 1)
			PlayerData._send_notification("please uninstall it if you wish to place off-canvas", 1)
			return false

	var offsets = []
	if two and grid == 0:
		shoulddel = true
		print(dir)
		vertical = false
		if imgy > imgx:
			vertical = true
		if vertical:
			match dir:
				"down":
					offsets = [
						Vector3(0 , 0, _offset),
						Vector3(0 , 0, - _offset),
					]
				"up":
					offsets = [
						Vector3(0 , 0, - _offset),
						Vector3(0 , 0, _offset),
					]
				"right":
					offsets = [
						Vector3(- _offset, 0, 0),
						Vector3(_offset, 0, 0),
					]
				"left":
					offsets = [
						Vector3(_offset, 0, 0),
						Vector3( - _offset, 0, 0),
					]
		else:
			match dir:
				"down":
					offsets = [
						Vector3(_offset, 0, 0),
						Vector3( - _offset, 0, 0),
					]
				"up":
					offsets = [
						Vector3( - _offset, 0, 0),
						Vector3(_offset, 0, 0),
					]
				"right":
					offsets = [
						Vector3(0 , 0, - _offset),
						Vector3(0 , 0, _offset),
					]
				"left":
					offsets = [
						Vector3(0 , 0, _offset),
						Vector3(0 , 0, - _offset),
					]
		for offset in offsets:
			var canvas_pos = pos + offset
			canvas_pos.y -= 0.008699999999999999
			var new_canvas_id = _Network._sync_create_actor("canvas", canvas_pos, current_zone)
			_canvas_id.append(new_canvas_id)
			print("Created new canvas at ", canvas_pos)
			var chalknode = _actor_man._get_actor_by_id(new_canvas_id)
			if chalknode:
				var smutnode = chalknode.get_node("chalk_canvas")
				_grid.append(smutnode.get_node("GridMap"))
				_tile.append(smutnode.get_node("Viewport/TileMap"))
			else:
				print("Failed to retrieve chalknode for canvas at ", canvas_pos)
	else:
		two = false
		var canvas_pos = Vector3.ZERO + pos
		if grid == 0:
			shoulddel = true
			canvas_pos.y -= 0.008699999999999999
			var new_canvas_id = _Network._sync_create_actor("canvas", canvas_pos, current_zone)
			_canvas_id.append(new_canvas_id)
			print("spawning at ", current_zone)
			print("Created new canvas at ", canvas_pos)
			_Chalknode = _actor_man._get_actor_by_id(new_canvas_id)
			if _Chalknode:
				_Smutnode = _Chalknode.get_node("chalk_canvas")
				_grid.append(_Smutnode.get_node("GridMap"))
				_tile.append(_Smutnode.get_node("Viewport/TileMap"))
			else:
				print("Failed to retrieve chalknode")
		elif grid == 1:
			game_canvas_id = 0
			_Chalknode = get_tree().current_scene.get_node_or_null("Viewport/main/map/main_map/zones/main_zone/chalk_zones/chalk_canvas")
			if _Chalknode:
				game_grid = _Chalknode.get_node("GridMap")
				game_tile = _Chalknode.get_node("Viewport/TileMap")
				print("using canvas ", grid)
			else:
				print("Failed to retrieve chalknode")
		elif grid == 2:
			game_canvas_id = 1
			_Chalknode = get_tree().current_scene.get_node_or_null("Viewport/main/map/main_map/zones/main_zone/chalk_zones/chalk_canvas2")
			if _Chalknode:
				game_grid = _Chalknode.get_node("GridMap")
				game_tile = _Chalknode.get_node("Viewport/TileMap")
			else:
				print("Failed to retrieve chalknode")
		elif grid == 3:
			game_canvas_id = 2
			_Chalknode = get_tree().current_scene.get_node_or_null("Viewport/main/map/main_map/zones/main_zone/chalk_zones/chalk_canvas3")
			if _Chalknode:
				game_grid = _Chalknode.get_node("GridMap")
				game_tile = _Chalknode.get_node("Viewport/TileMap")
			else:
				print("Failed to retrieve chalknode")
		elif grid == 4:
			game_canvas_id = 3
			_Chalknode = get_tree().current_scene.get_node_or_null("Viewport/main/map/main_map/zones/main_zone/chalk_zones/chalk_canvas4")
			if _Chalknode:
				game_grid = _Chalknode.get_node("GridMap")
				game_tile = _Chalknode.get_node("Viewport/TileMap")
			else:
				print("Failed to retrieve chalknode")
	display_image(file_path, pos)
	_chalk_send()

func check_image_resolution(file_path, pos):
	dir = _get_player_facing_direction()
	var file = File.new()
	isgif = false

	if not file.file_exists(file_path):
		push_error("File does not exist: %s" % file_path)
		return

	if file.open(file_path, File.READ) != OK:
		push_error("Failed to open file: %s" % file_path)
		return

	if not file.eof_reached():
		var size_line = file.get_line().strip_edges()
		var size_parts = size_line.split(",")
		imgx = size_parts[0].to_float()
		imgy = size_parts[1].to_float()
		if size_parts[2] == "gif":
			isgif = true

			print("its a gif")
			_framecount = size_parts[3].to_int()
			print(_framecount)
			_framedelay_ = size_parts[4].to_int()
			processing = false
			another = false

		var ogaboga = false
		if Input.is_key_pressed(KEY_1):
			pos = Vector3(48.571999, 0, - 51.041)
			PlayerData._send_notification("Spawning at canvas 1!", 0)
			ogaboga = true
		elif Input.is_key_pressed(KEY_2):
			pos = Vector3(69.57199900000001, 0, - 54.952999)
			PlayerData._send_notification("Spawning at canvas 2!", 0)
			ogaboga = true
		elif Input.is_key_pressed(KEY_3):
			pos = Vector3( - 54.7896, 0, - 115.719002)
			PlayerData._send_notification("Spawning at canvas 3!", 0)
			ogaboga = true
		elif Input.is_key_pressed(KEY_4):
			pos = Vector3( - 25.781099, 0, - 34.5681)
			PlayerData._send_notification("Spawning at canvas 4!", 0)
			ogaboga = true
		if ray_detector and not ogaboga:
			var ray_position = pos + Vector3(0, 20, 0)
			ray_detector.detect_collision_at(ray_position)
			yield (get_tree().create_timer(1.0 / 80.0), "timeout")
			ray_detector.detect_collision_at(ray_position)
			yield (get_tree().create_timer(1.0 / 80.0), "timeout")
			ray_detector.detect_collision_at(ray_position)
			yield (get_tree().create_timer(1.0 / 80.0), "timeout")
			ray_detector.detect_collision_at(ray_position)
			var ground_y = ray_detector.get_ground_y()
			if ground_y != null:
				pos.y = ground_y
			else:
				pos.y -= 0.9955000000000001
		if Input.is_key_pressed(KEY_DOWN):
			pos -= Vector3(0, 4, 0)
		else:
			if Input.is_key_pressed(KEY_UP):
				pos += Vector3(0, 4, 0)

		if (is_in_any_grid(pos) and current_zone == "main_zone") or _canvas_id.empty():
			gifdir = dir
			if imgx <= 20 and imgy <= 20:
				origin = pos
				two = false
				_spawn_canvas(origin, file_path)
			else:
				var _offset = null
				if imgy >= imgx:
					_offset = (imgy - 20) / 2
				else:
					_offset = (imgx - 20) / 2
				if _offset > 10 or _offset < 0:
					_offset = 10
				if new_canvas:
					origin = pos
					two = true
					_spawn_canvas(origin, file_path, _offset)
	file.close()

func display_image(file_path, pos):
	var file = File.new()
	if not file.file_exists(file_path):
		push_error("File does not exist: %s" % file_path)
		return
	if file.open(file_path, File.READ) != OK:
		push_error("Failed to open file: %s" % file_path)
		return

	var orientation = dir
	var base_offset = Vector3()
	if not isgif:
		match orientation:
			"down": base_offset = Vector3( - 0.5 * imgx, 0, 0.5 * imgy)
			"left": base_offset = Vector3( - 0.5 * imgy, 0, - 0.5 * imgx)
			"right": base_offset = Vector3(0.5 * imgy, 0, 0.5 * imgx)
			"up": base_offset = Vector3(0.5 * imgx, 0, - 0.5 * imgy)
			_:
				return
	else:
		match orientation:
			"down": base_offset = Vector3( - 0.5 * imgx, 0, - 0.5 * imgy)
			"left": base_offset = Vector3(0.5 * imgy, 0, - 0.5 * imgx)
			"right": base_offset = Vector3( - 0.5 * imgy, 0, 0.5 * imgx)
			"up": base_offset = Vector3(0.5 * imgx, 0, 0.5 * imgy)
			_:
				return

	file.get_line()
	var temparray = []
	while not file.eof_reached():
		var line = file.get_line().strip_edges()
		if line != "":
			var parts = line.split(",")
			if parts.size() == 3:
				var img_x = parts[0].to_float()
				var img_y = parts[1].to_float()
				var color = parts[2].to_int()
				if img_x == null or img_y == null or imgx == null or imgy == null:
					push_error("One or more variables (img_x, img_y, imgx, imgy) are null! Check file data.")
					continue
				var adjusted_position = Vector3()
				if not isgif:
					match orientation:
						"down": adjusted_position = Vector3(img_x, 0, - img_y)
						"left": adjusted_position = Vector3(img_y, 0, img_x)
						"right": adjusted_position = Vector3( - img_y, 0, - img_x)
						"up": adjusted_position = Vector3( - img_x, 0, img_y)
				else:
					match orientation:
						"down": adjusted_position = Vector3(img_x, 0, img_y)
						"left": adjusted_position = Vector3( - img_y, 0, img_x)
						"right": adjusted_position = Vector3(img_y, 0, - img_x)
						"up": adjusted_position = Vector3( - img_x, 0, - img_y)
				var world_position = pos + base_offset + adjusted_position + Vector3(0.02234, 0, 0.02234)
				if not grid == 0:
					temparray.append(world_position)
				_chalk_draw(world_position, color)
			else:
				push_error("Malformed line in file: %s" % line)
	if not grid == 0:
		if ctrlz_array.size() >= 32:
			ctrlz_array.remove(0)
		print("added from ", grid)
		ctrlz_array.append([temparray, game_grid, game_tile, game_canvas_id])

	base = base_offset + Vector3(0.01234, 0, 0.01234)
	_chalk_send()
	if isgif:
		match orientation:
			"up": base_offset = Vector3(0.5 * imgx, 0, 0.5 * imgy)
			"right": base_offset = Vector3( - 0.5 * imgy, 0, 0.5 * imgx)
			"left": base_offset = Vector3(0.5 * imgy, 0, - 0.5 * imgx)
			"down": base_offset = Vector3( - 0.5 * imgx, 0, - 0.5 * imgy)
			_:
				return
		if isready():
			newgif()

func toggle_playback(message = true):
	if Input.is_key_pressed(KEY_SHIFT):
		toggle_playback_mode(true)
		return

	if Input.is_key_pressed(KEY_CONTROL):
		reset_gif()
		return

	if not isgif or frames_path == "" or processing:
		if message:
			PlayerData._send_notification("No gif to play!", 1)
		return

	if not isready():
		PlayerData._send_notification("Still being processed!", 1)
		return

	if not another or frame_data_.size() != _framecount:
		newgif()
		return

	if playback_mode == PlaybackMode.MANUAL:
		# Step forward a single frame each time toggle_playback() is called
		if manual_frame_index >= frame_data_.size():
			manual_frame_index = 0
		_play_frame(manual_frame_index)
		manual_frame_index += 1
		return

	# Normal/Half/Slow mode toggle
	_playing = not _playing
	if not _playing:
		if message:
			PlayerData._send_notification("Playback Stopped", 1)
		return
	else:
		if message:
			PlayerData._send_notification("Playing!", 0)
		_play()


func reset_gif():
	PlayerData._send_notification("Set to frame 1", 0)
	_current_frame_index_ = 0
	manual_frame_index = 0
	_playing = false


func toggle_playback_mode(_message = false):
	var prior = playback_mode

	# Cycle the mode
	playback_mode = int(playback_mode) + 1
	if playback_mode > PlaybackMode.MANUAL:
		playback_mode = PlaybackMode.NORMAL

	# Optional text for the user
	var mode_names = ["Normal", "Half", "SLOW", "Manual"]
	if _message:
		PlayerData._send_notification("GIF mode changed to: " + mode_names[playback_mode], 0)

	if prior in [PlaybackMode.NORMAL, PlaybackMode.HALF, PlaybackMode.SLOW] and playback_mode == PlaybackMode.MANUAL:
		if _playing:
			_playing = false
		manual_frame_index = _current_frame_index_

	if prior == PlaybackMode.MANUAL and playback_mode in [PlaybackMode.NORMAL, PlaybackMode.HALF, PlaybackMode.SLOW]:
		_current_frame_index_ = manual_frame_index


func newgif():
	pass

func _play2():
	while _playing:
		if playback_mode == PlaybackMode.MANUAL:
			return
		else:
			# Draw the current frame
			_play_frame(_current_frame_index_)

			# Figure out how long to wait based on playback mode
			var delay = frame_delays_[_current_frame_index_] / 1000.0
			match playback_mode:
				PlaybackMode.NORMAL:
					yield (get_tree().create_timer(delay), "timeout")
				PlaybackMode.HALF:
					yield (get_tree().create_timer(delay * 2), "timeout")
				PlaybackMode.SLOW:
					yield (get_tree().create_timer(delay * 10), "timeout")

			# Move to next frame
			_current_frame_index_ += 1
			if _current_frame_index_ >= frame_data_.size():
				_current_frame_index_ = 0  # Loop around

			# If stopped mid-loop, exit
			if not _playing:
				return


func _play_frame2(frame_index):
	for pixel in frame_data_[frame_index]:
		var local_x = pixel["x"]
		var local_y = pixel["y"]
		var pixel_color = pixel["color"]

		var adjusted_position = Vector3()
		match dir:
			"up":
				adjusted_position = Vector3(-local_x, 0, -local_y)
			"right":
				adjusted_position = Vector3(local_y, 0, -local_x)
			"left":
				adjusted_position = Vector3(-local_y, 0, local_x)
			"down":
				adjusted_position = Vector3(local_x, 0, local_y)
			_:
				push_error("Invalid direction specified: %s" % dir)
				return

		var final_position = origin + base + adjusted_position
		_chalk_draw(final_position, pixel_color)
	_chalk_send()


func _chalk_draw(pos, color):
	if two:
		if vertical:
			match dir:
				"up":
					if pos.z >= origin.z:
						_map_and_draw(1, pos, color, send_load_2)
					elif pos.z < origin.z:
						_map_and_draw(0, pos, color, send_load)
				"down":
					if pos.z >= origin.z:
						_map_and_draw(0, pos, color, send_load)
					elif pos.z < origin.z:
						_map_and_draw(1, pos, color, send_load_2)

				"left":
					if pos.x >= origin.x:
						_map_and_draw(0, pos, color, send_load)
					elif pos.x < origin.x:
						_map_and_draw(1, pos, color, send_load_2)
				"right":
					if pos.x <= origin.x:
						_map_and_draw(0, pos, color, send_load)
					elif pos.x > origin.x:
						_map_and_draw(1, pos, color, send_load_2)
				_:
					pass
		else:
			match dir:
				"up":
					if pos.x >= origin.x:
						_map_and_draw(1, pos, color, send_load_2)
					elif pos.x < origin.x:
						_map_and_draw(0, pos, color, send_load)
				"down":
					if pos.x >= origin.x:
						_map_and_draw(0, pos, color, send_load)
					elif pos.x < origin.x:
						_map_and_draw(1, pos, color, send_load_2)

				"left":
					if pos.z >= origin.z:
						_map_and_draw(0, pos, color, send_load)
					elif pos.z < origin.z:
						_map_and_draw(1, pos, color, send_load_2)

				"right":
					if pos.z >= origin.z:
						_map_and_draw(1, pos, color, send_load_2)
					elif pos.z < origin.z:
						_map_and_draw(0, pos, color, send_load)
				_:
					pass
	else:
		_map_and_draw(0, pos, color, send_load)

func ctrlzgamecanvas():

	if ctrlz_array.size() == 0:
		PlayerData._send_notification("Nothing left to undo!", 1)
		return


	var last_entry = ctrlz_array[ - 1]


	var temparray = last_entry[0]
	var stored_game_grid = last_entry[1]
	var stored_game_tile = last_entry[2]
	var stored_canvas_id = last_entry[3]

	game_grid = stored_game_grid
	game_tile = stored_game_tile
	game_canvas_id = stored_canvas_id
	grid = game_canvas_id + 1

	for position in temparray:
		_map_and_draw(0, position, - 1, send_load)

	_chalk_send()

	ctrlz_array.pop_back()
	print("Undo action completed, last entry removed.")


func _map_and_draw(grid_idx, pos, color, load_array):
	if grid == 0:
		var result = _grid[grid_idx].world_to_map(pos - _grid[grid_idx].global_transform.origin)
		var p = Vector2(result.x, result.z) + Vector2(100, 100)
		_tile[grid_idx].set_cell(p.x, p.y, color)
		load_array.append([p, color])
		return
	else:
		var result = game_grid.world_to_map(pos - game_grid.global_transform.origin)
		var p = Vector2(result.x, result.z) + Vector2(100, 100)
		game_tile.set_cell(p.x, p.y, color)
		load_array.append([p, color])
		return

func _chalk_send():
	if grid == 0:
		if two:
			if send_load.size() > 0:
				Network._send_P2P_Packet({"type": "chalk_packet", "data": send_load.duplicate(), "canvas_id": _canvas_id[0]}, "all", 2, Network.CHANNELS.CHALK)
				send_load.clear()
			if send_load_2.size() > 0:
				Network._send_P2P_Packet({"type": "chalk_packet", "data": send_load_2.duplicate(), "canvas_id": _canvas_id[1]}, "all", 2, Network.CHANNELS.CHALK)
				send_load_2.clear()
		else:

			if send_load.size() > 0:
				Network._send_P2P_Packet({"type": "chalk_packet", "data": send_load.duplicate(), "canvas_id": _canvas_id[0]}, "all", 2, Network.CHANNELS.CHALK)
				send_load.clear()
	else:
		if send_load.size() > 0:
				Network._send_P2P_Packet({"type": "chalk_packet", "data": send_load.duplicate(), "canvas_id": game_canvas_id}, "all", 2, Network.CHANNELS.CHALK)
				send_load.clear()

func isready():
	var file = File.new()
	var config_path = key_handler.cnfg_path

	if file.open(config_path, File.READ) == OK:
		var data = file.get_as_text()
		file.close()

		var json_result = JSON.parse(data)
		if json_result.error == OK and typeof(json_result.result) == TYPE_DICTIONARY:
			var config_data = json_result.result

			if config_data["gif_ready"]:
				return true
			else:
				return false
