#pragma once

#include <map>
#include <memory>
#include <mutex>
#include "Camera.h"

class ViewCameraBinder
{
public:
	typedef std::map<Camera*, std::shared_ptr<Camera>> CameraPointerHolder;
	typedef std::map<int, std::shared_ptr<Camera>> ViewCameraMap;
	typedef std::shared_ptr<Camera> CameraPtr;

	class CameraIterator
	{
	public:
		CameraIterator();
		CameraIterator(const CameraIterator& other);
		CameraIterator& operator=(const CameraIterator& iterator);
		CameraIterator& operator++();
		CameraPtr& operator*() const;
		CameraPtr& operator->() const;
		bool operator==(const CameraIterator& other);
		bool operator!=(const CameraIterator& other);
		friend class ViewCameraBinder;
	private:
		CameraIterator(CameraPointerHolder::iterator iterator);
		CameraPointerHolder::iterator m_iterator;
	};

	class BindingIterator
	{
	public:
		BindingIterator();
		BindingIterator(const BindingIterator& other);
		BindingIterator& operator=(const BindingIterator& iterator);
		BindingIterator& operator++();
		std::pair<const int, CameraPtr>& operator*() const;
		std::pair<const int, CameraPtr>* operator->() const;
		bool operator==(const BindingIterator& other);
		bool operator!=(const BindingIterator& other);
		friend class ViewCameraBinder;
	private:
		BindingIterator(ViewCameraMap::iterator iterator);
		ViewCameraMap::iterator m_iterator;
	};

public:
	ViewCameraBinder();
	ViewCameraBinder(const ViewCameraBinder& other);
	ViewCameraBinder(ViewCameraBinder&& other);
	ViewCameraBinder& operator=(const ViewCameraBinder& other);
	void operator=(ViewCameraBinder&& other);

	void bind(int view_id, std::shared_ptr<Camera> camera);
	void unbind(int view_id, std::shared_ptr<Camera> camera);
	CameraPtr get_camera(int view_id);
	std::pair<ViewCameraBinder::CameraIterator, ViewCameraBinder::CameraIterator> get_all_cameras();
	std::pair<ViewCameraBinder::BindingIterator, ViewCameraBinder::BindingIterator> get_all_bindings();

	size_t camera_size();
	size_t view_size();
	void clear();

protected:
	std::mutex m_mutex;

	CameraPointerHolder m_pointer_holder;
	ViewCameraMap m_view_to_camera;
};
