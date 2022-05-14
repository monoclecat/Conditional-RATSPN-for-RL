import alr_envs

hand_envs = [
'HandReach-v0',
'HandManipulateBlockRotateZ-v0',
'HandManipulateBlockRotateZTouchSensors-v0',
'HandManipulateBlockRotateZTouchSensors-v1',
'HandManipulateBlockRotateParallel-v0',
'HandManipulateBlockRotateParallelTouchSensors-v0',
'HandManipulateBlockRotateParallelTouchSensors-v1',
'HandManipulateBlockRotateXYZ-v0',
'HandManipulateBlockRotateXYZTouchSensors-v0',
'HandManipulateBlockRotateXYZTouchSensors-v1',
'HandManipulateBlockFull-v0',
'HandManipulateBlock-v0',
'HandManipulateBlockTouchSensors-v0',
'HandManipulateBlockTouchSensors-v1',
'HandManipulateEggRotate-v0',
'HandManipulateEggRotateTouchSensors-v0',
'HandManipulateEggRotateTouchSensors-v1',
'HandManipulateEggFull-v0',
'HandManipulateEgg-v0',
'HandManipulateEggTouchSensors-v0',
'HandManipulateEggTouchSensors-v1',
'HandManipulatePenRotate-v0',
'HandManipulatePenRotateTouchSensors-v0',
'HandManipulatePenRotateTouchSensors-v1',
'HandManipulatePenFull-v0',
'HandManipulatePen-v0',
'HandManipulatePenTouchSensors-v0',
'HandManipulatePenTouchSensors-v1',
'HandReachDense-v0',
'HandManipulateBlockRotateZDense-v0',
'HandManipulateBlockRotateZTouchSensorsDense-v0',
'HandManipulateBlockRotateZTouchSensorsDense-v1',
'HandManipulateBlockRotateParallelDense-v0',
'HandManipulateBlockRotateParallelTouchSensorsDense-v0',
'HandManipulateBlockRotateParallelTouchSensorsDense-v1',
'HandManipulateBlockRotateXYZDense-v0',
'HandManipulateBlockRotateXYZTouchSensorsDense-v0',
'HandManipulateBlockRotateXYZTouchSensorsDense-v1',
'HandManipulateBlockFullDense-v0',
'HandManipulateBlockDense-v0',
'HandManipulateBlockTouchSensorsDense-v0',
'HandManipulateBlockTouchSensorsDense-v1',
'HandManipulateEggRotateDense-v0',
'HandManipulateEggRotateTouchSensorsDense-v0',
'HandManipulateEggRotateTouchSensorsDense-v1',
'HandManipulateEggFullDense-v0',
'HandManipulateEggDense-v0',
'HandManipulateEggTouchSensorsDense-v0',
'HandManipulateEggTouchSensorsDense-v1',
'HandManipulatePenRotateDense-v0',
'HandManipulatePenRotateTouchSensorsDense-v0',
'HandManipulatePenRotateTouchSensorsDense-v1',
'HandManipulatePenFullDense-v0',
'HandManipulatePenDense-v0',
'HandManipulatePenTouchSensorsDense-v0',
'HandManipulatePenTouchSensorsDense-v1',
]

mw_envs = [
    'AssemblyProMP-v2',  # gripping hard
    'DisassembleProMP-v2',

    'PickOutOfHoleProMP-v2',  # gripping easy
    'BinPickingProMP-v2',  # gripping hard
    'HammerProMP-v2',  # gripping and hitting
    'BoxCloseProMP-v2',  # gripping hard

    # button press task without wall and with wall
    'ButtonPressProMP-v2',
    'ButtonPressWallProMP-v2',

    'CoffeeButtonProMP-v2',
    'CoffeePullProMP-v2',
    'CoffeePushProMP-v2', # 10

    'PegInsertSideProMP-v2',

    # pick-up task with and without wall
    'PickPlaceWallProMP-v2',
    'PickPlaceProMP-v2',  # 13

    'StickPushProMP-v2',
    'StickPullProMP-v2',
]
i = 0
env = alr_envs.make('AssemblyProMP-v2', seed=1)
# render() can be called once in the beginning with all necessary arguments. To turn it of again just call render(None).
# You might need to set LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so
env.render()

state = env.reset()

for i in range(5):
    state, reward, done, info = env.step(env.action_space.sample()*100)

    # Not really necessary as the environments resets itself after each trajectory anyway.
    state = env.reset()
    # env.render()

