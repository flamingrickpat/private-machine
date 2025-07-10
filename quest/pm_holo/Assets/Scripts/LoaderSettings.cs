using UnityEngine;

public class SceneDataLoaderSettings : ScriptableObject
{
    // JSON scene file.
    [SerializeField] private TextAsset sceneJson;

    // Whether to load scene data on Start.
    [SerializeField] private bool loadSceneOnStart = true;

    [SerializeField]
    private RoomLoader.SceneDataSource sceneDataSource = RoomLoader.SceneDataSource.SceneApi;

    [SerializeField] private bool centerStaticMesh = true;

    public string SceneJson => sceneJson != null ? sceneJson.text : null;
    public bool LoadSceneOnStart => loadSceneOnStart;
    public RoomLoader.SceneDataSource SceneDataSource => sceneDataSource;
    public bool CenterStaticMesh => centerStaticMesh;
}