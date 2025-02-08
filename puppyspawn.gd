extends Spatial

onready var _PlayerData = get_node_or_null("/root/PlayerData")
onready var _OptionsMenu = get_node_or_null("/root/OptionsMenu")
onready var _Network = get_node_or_null("/root/Network")
onready var ray_detector_scene = preload("res://mods/PurplePuppy-Stamps/RayDetector.tscn")
onready var key_handler = get_tree().get_nodes_in_group("keys")[0]
var _keybinds_api = null
var cam = null
var ctrlz_array = []
var _hud = null
var _Chalknode = null
var _Smutnode = null
var _Canvas_Path = null
var _actor_man = null
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
var last_mouse_pos = Vector2()
var _current_frame_index = 0  
var send_load = []
var send_load_2 = []
var send_load_3 = []
var send_load_4 = []
var frame_data = []
var frame_delays = []
var imgx = null
var imgy = null
var last_mouse = null
var two = false
var dir = null
var gifdir = null
var origin = null
var isgif = false
var _framedelay = null
var _framecount = null
var _playing = false
var base = null
var processing = false
enum PlaybackMode{NORMAL, HALF, SLOW, MANUAL}
var playback_mode = PlaybackMode.NORMAL
var manual_frame_index = 0
var current_zone = "main_zone"
var img_override = false
var grid = 0
var shoulddel = false
var old_pid = 0
var another = false
enum BrushMode{
	PENCIL, 
	ERASER, 
}
var vertical = false
var brush_mode = BrushMode.PENCIL
var brush_size = 1
var brush_color = 0
var ray_detector = null



var img_path = null
var frames_path = null
var gui_path = null

var wait = false

var debug = true


func _ready():
	if debug:
		img_path = "res://mods/PurplePuppy-Stamps/game_data/current_stamp_data/stamp.txt"
		gui_path = "D:/Trash/GDWeave/mods/PurplePuppy-Stamps/imagePawcessor/imagePawcess.py"
		frames_path = "res://mods/PurplePuppy-Stamps/game_data/current_stamp_data/frames.txt"
	else:
		img_path = key_handler.img_path
		frames_path = key_handler.frames_path
		gui_path = key_handler.gui_path
	
	key_handler.connect("_play", self, "toggle_playback")
	key_handler.connect("spawn_stamp", self, "spawn_stamp")
	key_handler.connect("open_menu", self, "open_menu")
	key_handler.connect("_delete", self, "_delete")
	key_handler.connect("get_data", self, "get_data")
	_PlayerData.connect("_chalk_update", self, "_chalk_update")
	ray_detector = ray_detector_scene.instance()
	

func re_ready():
	_PlayerData = get_node_or_null("/root/PlayerData")
	_OptionsMenu = get_node_or_null("/root/OptionsMenu")
	_Network = get_node_or_null("/root/Network")
	_PlayerData.connect("_chalk_update", self, "_chalk_update")
	SceneTransition.connect("_finished", self, "re_ready")
	update_dynamic_nodes()


func _chalk_update(pos):
	last_mouse_pos = pos


func _delete(gif = false):
	pass
	if not update_dynamic_nodes():
		return 
	if _hud.using_chat and _hud:
		return 
	_framedelay = null
	processing = false
	_framecount = null
	_playing = false
	if not gif:
		isgif = false
	var delete = []
	var normal = true
	if key_handler.stupidincompatabilitydontpatchshitthatbreaksotherpeoplesmods:
		ctrlz()
		return
	for i in range(3):
		for actor in Network.OWNED_ACTORS:
			if actor.actor_type == "canvas":
				normal = false
				if not delete.has(actor):
					delete.append(actor)
	if normal:
		ctrlz()
	else:
		for actor in delete:
			_Player._wipe_actor(actor.actor_id)
		for i in range(5):
			for actor in delete:
				Network._send_actor_action(actor, "_wipe_actor", [actor.actor_id])
				yield (get_tree().create_timer(1.0 / 80.0), "timeout")
		delete.clear()
		_canvas_id.clear()
		_Chalknode = null
		_grid.clear()
		_tile.clear()
		shoulddel = false
		
	if gif:
		var file = File.new()

		if not file.file_exists(img_path):
			push_error("File does not exist: %s" % img_path)
			return 

		if file.open(img_path, File.READ) != OK:
			push_error("Failed to open file: %s" % img_path)
			return 

		if not file.eof_reached():
			var size_line = file.get_line().strip_edges()
			var size_parts = size_line.split(",")
			imgx = size_parts[0].to_float()
			imgy = size_parts[1].to_float()
			if size_parts[2] == "gif":
				isgif = true
				_framecount = size_parts[3].to_int()
				_framedelay = size_parts[4].to_int()
				processing = false
				_playing = false
				another = false
		
		
func open_menu():
	update_dynamic_nodes()
	if _hud and _hud.using_chat:
		return 
	
	if wait:
		return 
		
	wait = true
	
	var gui_dir = gui_path.get_base_dir()
	
	var command = "cmd.exe"
	var arguments = ["/C", "cd \"" + gui_dir + "\" && start \"\" \"" + gui_path + "\""]

	var pid = OS.execute(command, arguments, false)
	old_pid = pid
	
	if pid == 0:
		PlayerData._send_notification("Failed to launch my GUI qwq", 1)
		wait = false
		return 
		
	PlayerData._send_notification("Stamps menu launching externally!", 0)
	
	resetwait()
	
func resetwait():
	yield (get_tree().create_timer(1), "timeout")
	wait = false
	
		
func spawn_stamp():
	if not update_dynamic_nodes():
		return 
	if _hud and _hud.using_chat:
		return 
	check_image_resolution(img_path, last_mouse_pos)

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

func _get_player_facing_direction():
	if cam:
		var forward = cam.global_transform.basis.z.normalized()
		if abs(forward.x) > abs(forward.z):
			return "right" if forward.x > 0 else "left"
		else:
			return "down" if forward.z > 0 else "up"

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
			_framedelay = size_parts[4].to_int()
			processing = false
			_playing = false
			another = false

		if new_canvas:
			var ogaboga = false
			if Input.is_key_pressed(KEY_SHIFT):
				pos = _Player.global_transform.origin
				if ray_detector:
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
					ogaboga = true
			elif Input.is_key_pressed(KEY_SHIFT):
				pos = _Player.global_transform.origin
				if ray_detector:
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
					ogaboga = true
			elif Input.is_key_pressed(KEY_1):
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
			else:
				if Input.is_key_pressed(KEY_CONTROL):
					if shoulddel:
						_delete(true)
						yield (get_tree().create_timer(0.6), "timeout")
					pos = Vector3(154 + (imgy / 2), - 0.3, 1.4)
					dir = "left"
					PlayerData._send_notification("Spawning at dock!", 0)
					if imgx <= 20 and imgy <= 20:
						origin = pos
						two = false
						_spawn_canvas(origin, file_path)
						
							
						return 
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
							
								
						return 
						
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
			_playing = false
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
		else:
			_delete(true)
			yield (get_tree().create_timer(0.6), "timeout")
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
	
	if not another or frame_data.size() != _framecount:
		newgif()
		return
	
	if playback_mode == PlaybackMode.MANUAL:
		# Step forward a single frame each time toggle_playback() is called
		if manual_frame_index >= frame_data.size():
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
	_current_frame_index = 0
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
		manual_frame_index = _current_frame_index

	if prior == PlaybackMode.MANUAL and playback_mode in [PlaybackMode.NORMAL, PlaybackMode.HALF, PlaybackMode.SLOW]:
		_current_frame_index = manual_frame_index


func newgif():
	processing = true
	_playing = false
	
	frame_delays = []
	frame_data = []
	manual_frame_index = 0
	_current_frame_index = 0
	
	var file = File.new()
	if file.open(frames_path, File.READ) != OK:
		push_error("Failed to open file: %s" % frames_path)
		processing = false
		return
	
	var current_frame = null
	var current_delay = _framedelay if _framedelay != -1 else 100
	
	while not file.eof_reached():
		var line = file.get_line().strip_edges()
		
		if line.begins_with("frame,"):
			if current_frame != null:
				frame_data.append(current_frame)
				frame_delays.append(current_delay)
			
			current_frame = []
			var frame_parts = line.split(",")
			if _framedelay == -1 and frame_parts.size() > 2:
				current_delay = max(frame_parts[2].to_int(), 10)
			else:
				current_delay = _framedelay
		
		elif line != "":
			if current_frame == null:
				push_error("Pixel data encountered before frame start!")
				continue
			
			var pixel_parts = line.split(",")
			if pixel_parts.size() == 3:
				current_frame.append({
					"x": pixel_parts[0].to_float(),
					"y": pixel_parts[1].to_float(),
					"color": pixel_parts[2].to_int()
				})
	
	if current_frame != null:
		frame_data.append(current_frame)
		frame_delays.append(current_delay)
	
	file.close()
	another = true
	processing = false
	
	if frame_data.size() != _framecount:
		print("WTF")
	else:
		print("Frame data reset and loaded successfully!")


func _play():
	while _playing:
		if playback_mode == PlaybackMode.MANUAL:
			return
		else:
			# Draw the current frame
			_play_frame(_current_frame_index)
			
			# Figure out how long to wait based on playback mode
			var delay = frame_delays[_current_frame_index] / 1000.0
			match playback_mode:
				PlaybackMode.NORMAL:
					yield (get_tree().create_timer(delay), "timeout")
				PlaybackMode.HALF:
					yield (get_tree().create_timer(delay * 2), "timeout")
				PlaybackMode.SLOW:
					yield (get_tree().create_timer(delay * 10), "timeout")
			
			# Move to next frame
			_current_frame_index += 1
			if _current_frame_index >= frame_data.size():
				_current_frame_index = 0  # Loop around
			
			# If stopped mid-loop, exit
			if not _playing:
				return


func _play_frame(frame_index):
	for pixel in frame_data[frame_index]:
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

func ctrlz():
	
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


func get_data():
	if key_handler.in_game:
		var config_path = key_handler.cnfg_path
		var file = File.new()
		PlayerData._send_notification("Canvas data request received!", 0)
		var json_path = key_handler.data_path

		
		if file.open(json_path, File.WRITE) == OK:
			file.store_string("{}")
			file.close()
		else:
			PlayerData._send_notification("Failed to create or clear JSON file.", 1)
			_update_walky_talky_menu("Canvas data export failed")
			return 

		
		var canvas_nodes = {
			"Canvas 1": "Viewport/main/map/main_map/zones/main_zone/chalk_zones/chalk_canvas/Viewport/TileMap", 
			"Canvas 2": "Viewport/main/map/main_map/zones/main_zone/chalk_zones/chalk_canvas2/Viewport/TileMap", 
			"Canvas 3": "Viewport/main/map/main_map/zones/main_zone/chalk_zones/chalk_canvas3/Viewport/TileMap", 
			"Canvas 4": "Viewport/main/map/main_map/zones/main_zone/chalk_zones/chalk_canvas4/Viewport/TileMap"
		}

		var export_data = {}

		
		for canvas_name in canvas_nodes.keys():
			var canvas_path = canvas_nodes[canvas_name]
			var canvas = get_tree().current_scene.get_node_or_null(canvas_path)
			if canvas:
				var canvas_data = []
				for cell in canvas.get_used_cells():
					var color = int(canvas.get_cell(cell.x, cell.y))
					canvas_data.append(cell.x)
					canvas_data.append(cell.y)
					canvas_data.append(color)
				export_data[canvas_name] = canvas_data
			else:
				PlayerData._send_notification(canvas_name + " not found. Skipping.", 1)

		
		if file.open(json_path, File.WRITE) == OK:
			var json_string = JSON.print(export_data)
			file.store_string(json_string)
			file.close()
			yield (get_tree().create_timer(1.0 / 4.0), "timeout")
			PlayerData._send_notification("Data exported successfully!", 0)
			_update_walky_talky_menu("Canvas data exported!")
		else:
			yield (get_tree().create_timer(1.0 / 4.0), "timeout")
			PlayerData._send_notification("Data export failed", 1)
			_update_walky_talky_menu("Canvas data export failed")
	else:
		yield (get_tree().create_timer(1.0 / 4.0), "timeout")
		_update_walky_talky_menu("ur not in a lobby dummy")


func _update_walky_talky_menu(message):
	
	var config_path = key_handler.cnfg_path
	
	var file = File.new()

	
	if file.open(config_path, File.READ) == OK:
		var data = file.get_as_text()
		file.close()
		print("[Stamps] Read config file content:", data)
		
		
		var json_result = JSON.parse(data)
		if json_result.error == OK and typeof(json_result.result) == TYPE_DICTIONARY:
			var config_data = json_result.result

			
			config_data["walky_talky_menu"] = message
			print("[Stamps] Updated config_data:", config_data)

			
			if file.open(config_path, File.WRITE) == OK:
				var updated_json = JSON.print(config_data, "	")
				file.store_string(updated_json)
				file.close()
				print("[Stamps] walky_talky_menu successfully updated to:", message)
			else:
				print("[Stamps] Failed to open config file for WRITE operation.")
		else:
			print("[Stamps] Failed to parse config file. JSON Error:", json_result.error)
	else:
		print("[Stamps] Failed to open config file for READ operation.")
