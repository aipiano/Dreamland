#include "ViewCameraBinder.h"

ViewCameraBinder::ViewCameraBinder()
{
}

ViewCameraBinder::ViewCameraBinder(const ViewCameraBinder & other)
{
	m_pointer_holder = other.m_pointer_holder;
	m_view_to_camera = other.m_view_to_camera;
}

ViewCameraBinder::ViewCameraBinder(ViewCameraBinder && other)
{
	other.m_mutex.lock();
	m_pointer_holder = std::move(other.m_pointer_holder);
	m_view_to_camera = std::move(other.m_view_to_camera);
	other.m_mutex.unlock();
}

ViewCameraBinder & ViewCameraBinder::operator=(const ViewCameraBinder & other)
{
	m_mutex.lock();
	m_pointer_holder = other.m_pointer_holder;
	m_view_to_camera = other.m_view_to_camera;
	m_mutex.unlock();

	return *this;
}

void ViewCameraBinder::operator=(ViewCameraBinder && other)
{
	other.m_mutex.lock();
	m_mutex.lock();
	m_pointer_holder = std::move(other.m_pointer_holder);
	m_view_to_camera = std::move(other.m_view_to_camera);
	m_mutex.unlock();
	other.m_mutex.unlock();
}

void ViewCameraBinder::bind(int view_id, std::shared_ptr<Camera> camera)
{
	m_mutex.lock();
	Camera* ptr_model = camera.get();
	m_pointer_holder[ptr_model] = camera;
	m_view_to_camera[view_id] = camera;
	m_mutex.unlock();
}

void ViewCameraBinder::unbind(int view_id, std::shared_ptr<Camera> camera)
{
	m_mutex.lock();
	Camera* ptr_model = camera.get();
	if (m_view_to_camera[view_id] == camera)
		m_view_to_camera.erase(view_id);
	m_mutex.unlock();
}

std::shared_ptr<Camera> ViewCameraBinder::get_camera(int view_id)
{
	return m_view_to_camera[view_id];
}

std::pair<ViewCameraBinder::CameraIterator, ViewCameraBinder::CameraIterator> ViewCameraBinder::get_all_cameras()
{
	CameraIterator it_begin(m_pointer_holder.begin());
	CameraIterator it_end(m_pointer_holder.end());

	return std::make_pair(it_begin, it_end);
}

std::pair<ViewCameraBinder::BindingIterator, ViewCameraBinder::BindingIterator> ViewCameraBinder::get_all_bindings()
{
	BindingIterator it_begin(m_view_to_camera.begin());
	BindingIterator it_end(m_view_to_camera.end());
	
	return std::make_pair(it_begin, it_end);
}

size_t ViewCameraBinder::camera_size()
{
	return m_pointer_holder.size();
}

size_t ViewCameraBinder::view_size()
{
	return m_view_to_camera.size();
}

void ViewCameraBinder::clear()
{
	m_mutex.lock();
	m_view_to_camera.clear();
	m_pointer_holder.clear();
	m_mutex.unlock();
}


/////////////////////////////////////////////////////////
//				CameraIterator
////////////////////////////////////////////////////////
ViewCameraBinder::CameraIterator::CameraIterator()
{
}

ViewCameraBinder::CameraIterator::CameraIterator(const CameraIterator & other)
{
	m_iterator = other.m_iterator;
}

ViewCameraBinder::CameraIterator & ViewCameraBinder::CameraIterator::operator=(const CameraIterator & iterator)
{
	m_iterator = iterator.m_iterator;
	return *this;
}

ViewCameraBinder::CameraIterator & ViewCameraBinder::CameraIterator::operator++()
{
	++m_iterator;
	return *this;
}

std::shared_ptr<Camera> & ViewCameraBinder::CameraIterator::operator*() const
{
	return m_iterator->second;
}

std::shared_ptr<Camera>& ViewCameraBinder::CameraIterator::operator->() const
{
	return m_iterator->second;
}

bool ViewCameraBinder::CameraIterator::operator==(const CameraIterator & other)
{
	return m_iterator == other.m_iterator;
}

bool ViewCameraBinder::CameraIterator::operator!=(const CameraIterator & other)
{
	return !operator==(other);
}

ViewCameraBinder::CameraIterator::CameraIterator(CameraPointerHolder::iterator iterator)
{
	m_iterator = iterator;
}

/////////////////////////////////////////////////////////
//				BindingIterator
/////////////////////////////////////////////////////////
ViewCameraBinder::BindingIterator::BindingIterator()
{
}

ViewCameraBinder::BindingIterator::BindingIterator(const BindingIterator & other)
{
	m_iterator = other.m_iterator;
}

ViewCameraBinder::BindingIterator & ViewCameraBinder::BindingIterator::operator=(const BindingIterator & iterator)
{
	m_iterator = iterator.m_iterator;
	return *this;
}

ViewCameraBinder::BindingIterator & ViewCameraBinder::BindingIterator::operator++()
{
	++m_iterator;
	return *this;
}

std::pair<const int, std::shared_ptr<Camera>>& ViewCameraBinder::BindingIterator::operator*() const
{
	return *m_iterator;
}

std::pair<const int, std::shared_ptr<Camera>>* ViewCameraBinder::BindingIterator::operator->() const
{
	return m_iterator.operator->();
}

bool ViewCameraBinder::BindingIterator::operator==(const BindingIterator & other)
{
	return m_iterator == other.m_iterator;
}

bool ViewCameraBinder::BindingIterator::operator!=(const BindingIterator & other)
{
	return !operator==(other);
}

ViewCameraBinder::BindingIterator::BindingIterator(ViewCameraMap::iterator iterator)
{
	m_iterator = iterator;
}
